# Data

This directory is intended for the **raw ECG data used by the analysis pipeline**. The data used in this project was recorded with **CardioSecur Pro**. The expected structure is:

```text
data/
└── <experiment_name>/
    └── <experiment_session_number>/
        ├── <experimental_condition_1>/
        │   ├── <measurement>.xml
        │   └── <condition_metadata>.txt
        ├── <experimental_condition_2>/
        │   ├── <measurement>.xml
        │   └── <condition_metadata>.txt
        └── <experimental_condition_3>/
            ├── <measurement>.xml
            └── <condition_metadata>.txt
```

Here, `<experiment_session_number>` identifies the experimental session, while each `<experimental_condition>` represents a condition or phase of the experiment. The `.xml` and `.txt` files belong to the corresponding measurement.

The exact names and number of conditions may vary between datasets. **The names shown above are templates only** — preserve the original names and folder structure of your CardioSecur export.

> **Privacy:** ECG recordings are health data and may contain sensitive information. Do not commit real participant data to this repository. All contents of `data/` except this `README.md` are ignored by Git.
