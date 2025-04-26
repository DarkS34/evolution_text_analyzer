import pandas as pd

# File paths
MAP_FILE = "./SnomedCT_International_Edition_(Dependencia_EE_SNS)/SnomedCT_InternationalRF2_PRODUCTION_20240901T120000Z/Snapshot/Refset/Map/der2_iisssccRefset_ExtendedMapSnapshot_INT_20240901.txt"
DESCRIPTIONS_ES = "./SnomedCT_Spanish_Edition/SnomedCT_SpanishRelease-es_PRODUCTION_20240930T120000Z/Snapshot/Terminology/sct2_Description_SpanishExtensionSnapshot-es_INT_20240930.txt"
DESCRIPTIONS_EN = "./SnomedCT_International_Edition_(Dependencia_EE_SNS)/SnomedCT_InternationalRF2_PRODUCTION_20240901T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20240901.txt"
ICD_DATASET = "Diagnosticos_ES2024_TablaReferencia_30_06_2023_9096243130459179657.xlsx"

# Load and prepare mapping
mapping_df = pd.read_csv(MAP_FILE, sep="\t")
mapping_df = mapping_df[
    mapping_df['mapTarget'].notna() &
    (mapping_df['mapTarget'].str.strip() != '')
]
mapping_df['mapTarget'] = mapping_df['mapTarget'].str.upper()

# Load descriptions
desc_es_df = pd.read_csv(DESCRIPTIONS_ES, sep="\t")
desc_en_df = pd.read_csv(DESCRIPTIONS_EN, sep="\t")

# Filter active terms
desc_es = desc_es_df[
    (desc_es_df['languageCode'] == 'es') &
    (desc_es_df['active'] == 1)
][['conceptId', 'term']].rename(columns={'term': 'description_es'})

desc_en = desc_en_df[
    (desc_en_df['languageCode'] == 'en') &
    (desc_en_df['active'] == 1)
][['conceptId', 'term']].rename(columns={'term': 'description_en'})

# Merge both language descriptions (outer for completeness)
desc_all = pd.merge(desc_es, desc_en, on="conceptId", how="outer")

# Link descriptions to ICD codes via mapping
linked = pd.merge(
    mapping_df[['referencedComponentId', 'mapTarget']],
    desc_all,
    left_on='referencedComponentId',
    right_on='conceptId',
    how='inner'  # only SNOMED terms with ICD linkage
)

# Load the full official ICD-10 base
icd_base = pd.read_excel(
    ICD_DATASET, sheet_name="ES2024 Finales", usecols="A:B")
icd_base = icd_base.rename(columns={
    "Código": "icd_code",
    "Descripción": "description_es_normalized"
})
icd_base['icd_code'] = icd_base['icd_code'].str.upper()

# Join the base ICD with the SNOMED-linked descriptions
final_df = pd.merge(
    linked,
    icd_base,
    how="right",
    left_on="mapTarget",
    right_on="icd_code"
)

# Select and order columns
final_df = final_df[['icd_code', 'description_es_normalized',
                     'description_es', 'description_en']]

# MODIFICACIÓN: Filtrar solo los códigos ICD que comienzan con 'M'
final_df = final_df[final_df['icd_code'].str.startswith('M')]

# Sort for clarity
final_df = final_df.sort_values(by='icd_code')

# Export to CSV
final_df.to_csv("snomed_description_icd_normalized.csv",
                sep="\t", index=False, encoding="utf-8")
