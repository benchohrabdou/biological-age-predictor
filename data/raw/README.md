# Raw Data Directory

This folder is used to store raw NHANES `.xpt` files downloaded directly from the CDC servers. 

All `.xpt` files are excluded from Git version control (ignored in `.gitignore`) due to their size.

## Data Sources

The datasets are from the **NHANES August 2021–August 2023** survey cycle.

| Table Name | Description | Source URL |
| :--- | :--- | :--- |
| **DEMO_L.xpt** | Demographics | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt) |
| **BMX_L.xpt** | Body Measures | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt) |
| **BIOPRO_L.xpt** | Biochemistry | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.xpt) |
| **GHB_L.xpt** | Glycohemoglobin | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt) |
| **TCHOL_L.xpt** | Cholesterol | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt) |
| **SMQ_L.xpt** | Smoking Status | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt) |
| **CBC_L.xpt** | Complete Blood Count | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt) |
| **PAQ_L.xpt** | Physical Activity | [Link](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.xpt) |

## Ingestion
You do not need to download these files manually. They are downloaded automatically when you run:
```bash
python src/data_loader.py
```
