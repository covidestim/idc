The input file is a csv with one of two header rows:

```
state,date,cases,deaths
fips,date,cases,deaths
```

Where:
  - `state` is one of the fifty states plus "Puerto Rico" and "District of Columbia"
  - `fips` is a 5-character string matching /^[0-9]{5}$/
  - `cases` is an nonnegative integer
  - `deaths` is an nonnegative integer

The output file will be a CSV with one of two headers:

```
state,date,system_state
fips,date,system_state
```

Where:
  - `state` is one of the fifty states plus PR and DC
  - `fips` is a 5-character string matching /^[0-9]{5}$/
  - `system_state` is one of the following strings:
    - `NOMINAL`
    - `NONREPORTING`
    - `EXPECTED_DUMP`
    - `DUMP`
