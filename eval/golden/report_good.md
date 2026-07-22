# Golden gate — PASS ✅  (live, good prompt)

score 92% vs threshold 75%  (11/12 cases)

- abstain  : 1/1
- exec     : 8/8
- reference: 2/3

| case | kind | result | detail |
|---|---|---|---|
| g01_qualify | exec | ✅ | matched 2 row(s) |
| g02_unnest | exec | ✅ | matched 3 row(s) |
| g03_list_agg | exec | ✅ | matched 2 row(s) |
| g04_string_split | exec | ✅ | matched 1 row(s) |
| g05_regexp_extract | exec | ✅ | matched 1 row(s) |
| g06_star_exclude | exec | ✅ | matched 1 row(s) |
| g07_coalesce | exec | ✅ | matched 1 row(s) |
| g08_range_count | exec | ✅ | matched 1 row(s) |
| g09_pivot_ref | reference | ✅ | cited ['sql/statements/pivot.md'] |
| g10_asof_ref | reference | ❌ | cited none of ['guides/sql_features/asof_join.md'] |
| g11_secret_ref | reference | ✅ | cited ['configuration/secrets_manager.md'] |
| g12_abstain | abstain | ✅ | abstained |