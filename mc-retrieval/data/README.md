# 📇 Dataset Card: Minecraft Schematics

## 📖 Dataset Overview

This dataset contains 3D Minecraft structures (schematics) represented as flattened voxel arrays, along with rich metadata scraped from Planet Minecraft (titles, descriptions, tags, authors, and download links).

## 📁 File Information

- **File Name:** `data.parquet`
- **Format:** Apache Parquet
- **File Size:** 11.01 MB
- **Number of Records:** 8,328
- **Number of Features:** 19

*(Note: There is an updated version of this dataset, `data_with_voxel_names.parquet` (22.0 MB) featuring `voxel_name_data`, available on Kaggle: [minecraft-schematics-mvm](https://www.kaggle.com/datasets/farhanwew/minecraft-schematics-mvm?select=data_with_voxel_names.parquet))*

## 🛠️ Schema & Features

- **`url`**: `object`
- **`voxel_data`**: `object`
- **`title`**: `object`
- **`subtitle`**: `object`
- **`img`**: `object`
- **`user`**: `object`
- **`date`**: `object`
- **`bigImgs`**: `object`
- **`description`**: `object`
- **`tags`**: `object`
- **`diamondCount`**: `int64`
- **`views`**: `int64`
- **`downloads`**: `int64`
- **`comments`**: `float64`
- **`favorites`**: `int64`
- **`downloadLink`**: `object`
- **`finalDownloadLink`**: `object`
- **`thirdPartyDownloadLink`**: `object`
- **`youtubeId`**: `object`

## 📝 Notes

- `voxel_data` contains a flattened representation of a 32x32x32 3D array of Minecraft block IDs.
- List or dictionary-based metadata fields (like `tags` or `bigImgs`) were serialized into JSON strings to comply with Parquet's column constraints.
- Dataset was obtained from Romain Beaumont and reformatted into a parquet for ease of use. [^1]

[^1]: Beaumont, R. (2024). _minecraft-schematics-dataset_ [Data set]. GitHub. https://github.com/rom1504/minecraft-schematics-dataset
