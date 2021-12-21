'''
Input Data Classifier

Azure key must be passed as the environment variable AZURE_OCP_APIM_SUBSCRIPTION_KEY.

Usage:
  idc.py -o <path> --key <key> <input_path>
  idc.py -h | --help
  idc.py --version
  idc.py --test


Options:
  <input_path>  Path to a CSV of input data.
  -o <path>     Path to write output CSV to.
  --key <key>   Which geo type is being used ("state"/"fips")
  --test        run automated doctests
  -h --help     Show this screen.
'''
import sys
import os
import numpy
import numpy.linalg
import scipy.stats
from  scipy.stats import gamma
import http.client
import json
import urllib
import time
import datetime
import hmmlearn
from docopt import docopt
from itertools import groupby
from tqdm import tqdm

STATES=["NOMINAL", "NONREPORTING", "EXPECTED_DUMP", "DUMP"]


#We could use a library like pandas to read in the csv, but I'm trying to minimize library dependencies
def inputDataFromCSV(fName):
    inF=open(fName,'r')
    line=inF.readline().strip()
    if (line=='state,date,cases,deaths'):
        isFIPS=False
        region='state'
    elif (line=='fips,date,cases,deaths'):
        isFIPS=True
        region='fips'
    else:
        raise Exception('input csv header should be either state,date,cases,deaths or fips,date,cases,deaths, not ' + line)
    line=inF.readline()
    rows=[]
    while (line != ''):
        splts=line.strip().split(',')
        rows.append({region:str(splts[0]),
                     'date':str(splts[1]),
                     'cases':int(splts[2]),
                     'deaths':int(splts[3])})
        line=inF.readline()

    def keyDate(val):
        return val['date']

    def keyTract(val):
        return val[region]

    rows_sorted = sorted(rows, key=keyTract)
    rows_grouped = [sorted(list(it), key=keyDate) for k, it in groupby(rows_sorted, keyTract)]

    return rows_grouped


def inputDataToCSV(countyInput,outName,isFIPS=True):
    inpD=inputData(countyInput)
    outF=open(outName,'w')
    if (isFIPS):
        region='fips'
    else:
        region='state'
    outF.writelines(','.join([region,'date','cases','deaths']) + "\n"  )
    for row in inpD:
        outF.writelines(','.join([row[region],row['date'],"%d"%(row['cases']),("%d"%row['deaths']) + "\n" ]))
    outF.close()

def inputData(countyInput):
    """
    >>> nh=inputData('09009')
    >>> nh[0]
    {'fips': '09009', 'date': '2020-03-14', 'cases': 4, 'deaths': 0}
    """

    state = countyInput

    isCounty=False
    if (len(state)==5):
        if (state.isnumeric()):
            isCounty=True

    if (isCounty):
        geoColumn="fips"
    else:
        geoColumn="state"
    covidestimAPIBase  = "api.covidestim.org"
    if (isCounty):
        covidestimAPITable ="latest_inputs"
    else:
        covidestimAPITable = "latest_state_inputs"
    #Query latest input data for a state
    covidestimAPIUrl   = "/" + covidestimAPITable +"?" + geoColumn + "=eq." + state
    conn=http.client.HTTPSConnection(covidestimAPIBase)
    #print(covidestimAPIUrl)
    conn.request("GET",covidestimAPIUrl) 
    resp=conn.getresponse()
    rawStr=resp.read().decode("utf-8")
    return json.loads(rawStr)


def loadAllGeocodes():
    inF=open('all-geocodes.csv','r')

    line=inF.readline()
    line=inF.readline()
    fips=[]
    while (line != ''):
        splts=line.strip().split(',')
        region=splts[-1]
        if (region.find(' County') != -1):
            stateFIPS=splts[1]
            countyFIPS=splts[2]
            fips.append(stateFIPS + countyFIPS)
        try:
            line=inF.readline()
        except:
            pass
            #print('exception:')
            #print(line)
    return fips


def doForFIPSRange(start,end):
    fips=loadAllGeocodes()
    paramDict={'maxAnomalyRatio':0.075,'sensitivity':30}

    for f in fips[start:end]:
        print(f)
        inputDat=inputData(f)
        anom=anomalyDetector(inputDat,paramDict)
        if (len(anom) < 1000):
            print('throttled')
        outF=open('anom_' + f,'w')
        outF.writelines(anom)
        outF.close()
        #time.sleep(3)

def anomalyDetector(inputData,paramDict):
    """
    >>> azureOutput=anomalyDetector(inputData('09009'),{'maxAnomalyRatio':0.075,'sensitivity':30})
    >>> numpy.where(azureOutput['isAnomaly'])[0]
    array([ 58, 240, 255, 261, 264, 268, 272, 275, 289, 303, 310, 317, 325,
           331, 394])
    """
    maxAnomalyRatio=paramDict['maxAnomalyRatio']
    sensitivity=paramDict['sensitivity']
    azureAPIBase = "covidestim.cognitiveservices.azure.com"

    azureAPIAnomalyBase ="/anomalydetector/v1.0/timeseries/entire/detect"

    #azureAPIAnomalyBase ="/anomalydetector/v1.0/timeseries/entire/detect"

    nonZeroIndices=set()

    nonZeroRows=[]

    numDays=len(inputData)

    indexMap=dict()

    nzero = len(list(filter(lambda x: x['cases'] < 1, inputData)))
    for i in range(numDays):
        if ((inputData[i]['cases'] > 0) | (nzero/numDays > 0.3)):
            indexMap[i]=len(nonZeroIndices)
            nonZeroIndices.add(i)
            nonZeroRows.append(inputData[i])

    azureHeaders = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": os.environ["AZURE_OCP_APIM_SUBSCRIPTION_KEY"] }
    jsonSeriesStr='{ "series" :[ '

    #TODO deaths???
    timepointList=",".join(['{ "timestamp" : "%sT00:00:00Z", "value": %d }'%(row['date'],row['cases']) for row in nonZeroRows])

    jsonSeriesStr += timepointList + "], "
    jsonSeriesStr += '"maxAnomalyRatio": %f, '%maxAnomalyRatio
    jsonSeriesStr += '"sensitivity": %d, '%sensitivity
    jsonSeriesStr += ' "granularity": "daily", '
    jsonSeriesStr += ' "period": 7 }'

    jsonSeries=json.dumps(jsonSeriesStr)

    conn=http.client.HTTPSConnection(azureAPIBase)
    conn.request("POST",azureAPIAnomalyBase,body=jsonSeriesStr,headers=azureHeaders)
    #print('waiting') 
    resp=conn.getresponse()
    #print('done waiting')
    rawStr=resp.read().decode()
    response=json.loads(rawStr)

    if (response.get('message','').find('Ratio of missing points should be less') != -1):
        print(response['message'])
        print(inputData[0])
        print(timepointList)
        raise Exception('not enough nonzero case dates for Azure')

    if (response.get('message','').find("The 'series' field must have at least") != -1):
        print(response['message'])
        raise Exception('not enough case dates for Azure')

    respDct=dict()
    FIELDS=['expectedValues', 'isAnomaly', 'isNegativeAnomaly', 'isPositiveAnomaly', 'lowerMargins', 'upperMargins']
    for i in range(numDays):
        if (i in nonZeroIndices):
            j=indexMap[i]
            for field in FIELDS:
                try:
                    respDct[field]=respDct.get(field,[]) + [response[field][j]]
                except:
                    print(response)
                    import pdb; pdb.set_trace()
        else:
            for field in FIELDS:
                respDct[field]=respDct.get(field,[]) + [None]
    return respDct
    

def movingAverage(values,N):
    '''
    >>> movingAverage(numpy.array([5,3,7,9,3,8,4,6]),3)
    array([       nan,        nan, 5.        , 6.33333333, 6.33333333,
           6.66666667, 5.        , 6.        ])

    '''
    sm=0 
    means=numpy.array(len(values)*[numpy.nan])

    theMin=numpy.min([N-1,len(values)])
    
    for i in range(theMin):
        sm += values[i]
    
    for j in range(theMin,len(values)):
        sm += values[j]
        means[j]=sm/N
        sm -= values[j - N + 1]
        
    return means

 
def predictNextObservation(arr,movAvgLen):
    '''
    >>> predictNextObservation(numpy.array([4,4,5,5,5,5,6,7,7.5,8,8.5,9,9.5]),5)
    array([  nan,   nan,   nan,   nan,  5.5 ,  5.4 ,  5.8 ,  7.1 ,  8.2 ,
            8.95,  9.2 ,  9.5 , 10.  ])

    '''
    length=len(arr)

    subsets=zip(range(length-movAvgLen + 1),
                range(movAvgLen,length+1))

    res=numpy.array((length)*[numpy.nan])
    for i in range(movAvgLen,length + 1):
        #print('i is',i)
        X=numpy.array([range(i-movAvgLen,i),movAvgLen*[1]],dtype=numpy.float64)
        y=arr[i-movAvgLen:i]
        #print('y',y)
        slope,intercept=numpy.linalg.lstsq(X.T, y, rcond=None)[0]
        res[i-1]=slope*i + intercept
    return res
         
 
def ppoisson(x,lambd):
    """

    >>> ppoisson(3,5.5)
    0.34229595583459105
    """
    return 1 - scipy.stats.gamma.cdf(lambd,x+1,1)


class PoissonInfo(object):

    def __init__(self,observed,predicted,quantile,toolow,toohigh):
        self.observed=observed
        self.predicted=predicted
        self.quantile=quantile
        self.toohigh=toohigh
        self.toolow=toolow
        

def poissonBounds(obs,nDaysAvg=7,quantiles=(0.001,0.9999999999)):#[qLo,qHi]):
    avg=movingAverage(obs,nDaysAvg)

    qLo,qHi=quantiles
    regressed=predictNextObservation(obs,nDaysAvg)

    def overrideTest(theObs,thePred):
        if (theObs==0 and thePred==0):
            return True
        if (theObs < 10 and thePred==0):
            return True
        return False

    returnList=[]
    #HACk, reviist
    returnList.append(PoissonInfo(0,0,0.5,False,False))
    for i in range(1,len(obs)):
        #the poisson im
        quantile=ppoisson(obs[i],
                          #the gamma implementation in the library
                          #I'm using has a different convention than
                          #what Marcus used, so that's why I need the +1
                          regressed[i-1]+1)
        
        if (overrideTest(obs[i],regressed[i-1])):
            toolow=True
            toohigh=True
        else:
            toolow=quantile < qLo
            toohigh=quantile > qHi
        returnList.append( PoissonInfo(obs[i],regressed[i-1],quantile,toolow,toohigh))
    return returnList
 

#TODO get this from the UI once we hook everything up
NDAYSAVG2=14


class Observation(object):

    def __init__(self,isZero,isPoissonAnomalous,isAzureAnomalous,isWeekend):

        self.isZero=isZero
        self.isPoissonAnomalous=isPoissonAnomalous
        self.isAzureAnomalous=isAzureAnomalous
        self.isWeekend=isWeekend
        
def observationFromTS(inputData):
    ts=[row['cases'] for row in inputData]
    dates=[row['date'] for row in inputData]

    def toDateObj(dateStr):
        splts=dateStr.strip().split('-')
        return datetime.date(int(splts[0]),int(splts[1]),int(splts[2]))

    dateObjs=[toDateObj(dStr) for dStr in dates]

    def isWeekend(dateObj):
        return dateObj.weekday() > 4


    isWeekendArray=[isWeekend(dObj) for dObj in dateObjs]
    
    poissonTestArray=poissonBounds(ts,NDAYSAVG2)

    azureTestArray=anomalyDetector(inputData,
                                   {'maxAnomalyRatio':0.075,
                                    'sensitivity':30})
 
    az=[azureTestArray['isAnomaly'][i] for i in range(len(azureTestArray['isAnomaly']))]

    
    def isZero(obs,predicted):
        if (obs <= 0):
            return True
        if (obs < numpy.floor(0.02*predicted)):
            return True
        return False

    def makeObservation(poissonArray,azureArray,isWeekendArray,i):

        pt=poissonArray[i]

        isPoissonAnomalous=pt.toolow or pt.toohigh
        if (azureArray['isAnomaly'][i]==None):
            isAzureAnomalous=False
        else:
            isAzureAnomalous=azureArray['isAnomaly'][i] and not azureArray['isNegativeAnomaly'][i]
       
        return Observation(isZero(pt.observed,pt.predicted),isPoissonAnomalous,
                           isAzureAnomalous,isWeekendArray[i])
 
    def symbolToNum(obs):
        num=0

        if (obs.isZero):
            num += 8
        if (obs.isPoissonAnomalous):
            num += 4
        if (obs.isAzureAnomalous):
            num += 2
        if (obs.isWeekend):
            num += 1
        return num

    observations=[makeObservation(poissonTestArray,azureTestArray,isWeekendArray,i) for i in range(len(poissonTestArray))]
     
    return [symbolToNum(obs) for obs in observations][1:]



PI=numpy.array([0.67,0.22,0,0.11])

A=numpy.array([[0.7,0.2,0.02,0.08],
               [0.1,0.3,0.57,0.03],
               [0.75,0.17,0.07,0.01],
               [0.9,0.06,0.01,0.03]])


E=numpy.array([[0.35,0.15,0.01,0.02,0.08,0.06,0.01,0,0.05,0.05,0.01,0.01,0.03,0.02,0.07,0.07],
   [0,0,0,0,0,0,0,0,0.02,0.06,0.05,0.17,0.16,0.47,0.02,0.05],
   [0.01,0.02,0.15,0.03,0.57,0.09,0.07,0.07,0,0,0,0,0,0,0,0],
   [0.01,0.01,0.07,0.06,0.06,0.04,0.49,0.26,0,0,0,0,0,0,0,0] ])



# from https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm/9730083
# Not currently used but keeping it around in case we want to swap it in
def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = numpy.empty((K, T), 'd')
    T2 = numpy.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = numpy.max(T1[:, i - 1] * A.T * B[numpy.newaxis, :, y[i]].T, 1)
        T2[:, i] = numpy.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = numpy.empty(T, 'B')
    x[-1] = numpy.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


#From https://ben.bolte.cc/viterbi. Not currently used but keeping around for now in case we want to swap it in
def step(mu_prev,
         emission_probs,
         transition_probs,
         observed_state):
    """Runs one step of the Viterbi algorithm.
    
    Args:
        mu_prev: probability distribution with shape (num_hidden),
            the previous mu
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        observed_state: the observed state at the current step
    
    Returns:
        - the mu for the next step
        - the maximizing previous state, before the current state,
          as an int array with shape (num_hidden)
    """
    
    pre_max = mu_prev * transition_probs.T
    max_prev_states = numpy.argmax(pre_max, axis=1)
    max_vals = pre_max[numpy.arange(len(max_prev_states)), max_prev_states]
    mu_new = max_vals * emission_probs[:, observed_state]
    
    return mu_new, max_prev_states

def viterbi_v2(emission_probs,
            transition_probs,
            start_probs,
            observed_states):
    """Runs the Viterbi algorithm to get the most likely state sequence.
    
    Args:
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        start_probs: the initial probabilies for each state, with shape
            (num_hidden)
        observed_states: the observed states at each step
    
    Returns:
        - the most likely series of states
        - the joint probability of that series of states and the observed
    """
    
    # Runs the forward pass, storing the most likely previous state.
    mu = start_probs * emission_probs[:, observed_states[0]]
    all_prev_states = []
    for observed_state in observed_states[1:]:
        mu, prevs = step(mu, emission_probs, transition_probs, observed_state)
        all_prev_states.append(prevs)
    
    # Traces backwards to get the maximum likelihood sequence.
    state = numpy.argmax(mu)
    sequence_prob = mu[state]
    state_sequence = [state]
    for prev_states in all_prev_states[::-1]:
        state = prev_states[state]
        state_sequence.append(state)

    #print('used new one')
    return state_sequence[::-1], sequence_prob


#Intended to be a line-by-line translation of the Javascripot implementation in the Observable notebook (hence the 'FromJS' name).
def viterbiFromJS(start_probability,
                  emission_probability,
                  transition_probability,
                  observations):
    ln = numpy.log
  
    V = [{}]
    path = {}

    numStates=len(start_probability)
    
    #Initialize base cases (t == 0)
    for state in range(numStates):
        #state = states[i];
        V[0][state] = ln(start_probability[state]) + ln(emission_probability[state][observations[0]])
        path[state] = [state]
    

    #Run Viterbi for t > 0
    for t in range(1,len(observations)):
    
        V.append({})
        newpath = {}
         
        for state in range(numStates):
            
            mx = [-numpy.inf,None]
            
            for state0 in range(numStates):
             
                #Calculate the probablity
                calc = V[t-1][state0] + ln(transition_probability[state0][state]) + ln(emission_probability[state][observations[t]])
                #This line is a strict > comparison in the javascript version but
                # >= seems to be required here due to differing behaviors involving nans, although I should revisit and double-check this
                if(calc >= mx[0]):
                    mx = [calc,state0]
                
                
            V[t][state] = mx[0]
            
            newpath[state] = path[mx[1]] + [state]
            
        path = newpath

    mx = [-numpy.inf,None]
    for state in range(numStates):
        #var state = data.states[i];
        calc = V[len(observations)-1][state]
        if(calc > mx[0]):
            mx = [calc,state]

    return [mx[0], path[mx[1]]]

#This is the overall end-to-end function from input csv to output csv
def classifyAndDumpCSV(inFileName,outFileName,isFIPS):
    outF=open(outFileName,'w')
    allObservations=inputDataFromCSV(inFileName)

    if (isFIPS):
        region='fips'
    else:
        region='state'

    colNames=[region,'date','system_state']
    outF.writelines(','.join(colNames) + '\n')

    for observations in tqdm(allObservations):
        res=runViterbi(observations)
        regionList=[o[region] for o in observations[1:]]
        dateList=[o['date'] for o in observations[1:]]
        caseList=[o['cases'] for o in observations[1:]]
        stateList=[STATES[r] for r in res[1]]
        for i in range(len(dateList)):
            outF.writelines('%s,%s,%s\n'%(regionList[i],dateList[i],stateList[i]))

    outF.close()


def runViterbi(observations):
    detailedObs=observationFromTS(observations)

    #I'm leaving in the calls to the other viterbit implementations, commented out, in case we want to
    #examine the differences between implementations
    #result=viterbi(detailedObs,A,E,PI)
    #result=viterbi_v2(E,A,PI,detailedObs)

    result=viterbiFromJS(PI,E,A,detailedObs)
    return result

                                      
def runTests():
    import doctest
    print('running doctest')
    doctest.testmod()

if __name__=='__main__':
    if (os.environ.get("AZURE_OCP_APIM_SUBSCRIPTION_KEY",'')==''):
        print('environment variable AZURE_OCP_APIM_SUBSCRIPTION_KEY must be set')
        sys.exit(1)
    args=docopt(__doc__)

    if (args['--test']):
        runTests()
    if (args['-o']):
        outPath=args['-o']
        fipsOrState=args['--key']
        if (not (fipsOrState in ['state','fips'])):
            print('key must either equal "fips" or "state"')
            sys.exit(1)
        isFIPS=fipsOrState=='fips'
        classifyAndDumpCSV(args['<input_path>'],outPath,isFIPS)


    #runTests()
 
