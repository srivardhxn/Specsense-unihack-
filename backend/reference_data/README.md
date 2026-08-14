# Controlled Vocabulary Reference Data

Drop real reference files here to activate compliance validation
(manufacturer/brand names, units, and attribute values checked against
Unilog's actual approved lists, per the Solution Guide's requirements).

**Nothing in this folder is fabricated data.** If a file below isn't
present, that specific validation stays OFF and fields pass through
unchanged — the pipeline is honest about what it has and hasn't
checked, rather than pretending to validate against a list it doesn't
have.

## Files this folder looks for (any subset is fine)

### `manufacturer_brand.csv`
From `UniCat_Manufacturer_and_Brand_List.xlsx`. Export the relevant
columns as CSV with headers exactly:
```
MANUFACTURER_NAME,BRAND_NAME
```

### `uom_standards.csv`
From `Unilog_Master_UOM_Standards_Abbreviations_and_Terms.xlsx`
(Sheet 1). Export as CSV with headers:
```
Measurement_Type,Approved_Abbreviation
```

### `fittings_lov.csv`
From `Fittings_LOV.xlsx` or the Fittings rows of
`Unicat_Lov_v1_0_Updated_With_Remarks.xlsx`. Export as CSV with
headers:
```
Attribute_Label,Attribute_Value,Normalized_Value
```

## After adding files

Restart the server. Check the startup logs — each file that loaded
successfully prints how many entries it found:

```
[vocabulary] manufacturer/brand list ACTIVE: 27431 entries loaded
[vocabulary] UOM standards ACTIVE: 503 mappings loaded
[vocabulary] Fittings LOV ACTIVE: 1247 value mappings loaded
```

Or check `GET /api/vocabulary/status` at any time while the server is
running.

## If the real files have different column names

The loader in `backend/services/vocabulary.py` expects the exact
headers above. If your exported CSV has different column names, either
rename the columns in your CSV, or edit the `_load_csv_rows` calls in
`vocabulary.py` to match your actual headers.
