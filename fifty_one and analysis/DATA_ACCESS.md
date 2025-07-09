# Accessing the Prawn Research Datasets

This guide explains how to access and work with our research datasets hosted on Google Drive.

## Dataset Links

1. **ImageJ Measurements Dataset (~20GB)**
   - [Link to folder]
   - Contains prawn measurement images from three pond types

2. **Molt/Exuviae Dataset (~10GB)**
   - [Link to folder]
   - Contains original and processed molt images

3. **Drone Detection Dataset (~15GB)**
   - [Link to folder]
   - Contains aerial imagery and detection results

## Access Methods

### 1. Google Drive Desktop App (Recommended)

1. **Installation**:
   - Download from [Google Drive for Desktop](https://www.google.com/drive/download/)
   - Install and sign in with your Google account
   - Request access to the shared folders

2. **Configuration**:
   ```bash
   # Example directory structure
   ~/Google Drive/
   ├── imagej_measurements/
   ├── molt_exuviae/
   └── drone_detection/
   ```

3. **Usage**:
   - Files sync automatically
   - Work with local copies
   - Changes sync back to Drive

### 2. Browser Download

1. Navigate to the shared folder
2. Select files/folders to download
3. Use "Download" button
4. Extract to your project directory

### 3. Programmatic Access (Python)

1. **Setup**:
   ```bash
   pip install gdown
   ```

2. **Download Example**:
   ```python
   import gdown
   
   # Download a specific file
   url = "your_google_drive_file_url"
   output = "local_filename"
   gdown.download(url, output, quiet=False)
   
   # Download entire folder
   url = "your_google_drive_folder_url"
   gdown.download_folder(url, quiet=False)
   ```

## Best Practices

1. **Storage Management**:
   - Total dataset size: ~45GB
   - Ensure sufficient local storage
   - Consider selective syncing

2. **Working with Large Files**:
   - Use streaming when possible
   - Process in batches
   - Clean up temporary files

3. **Backup Considerations**:
   - Google Drive maintains versions
   - Local backups recommended
   - Document any modifications

## Troubleshooting

1. **Access Issues**:
   - Ensure you have the correct permissions
   - Check your Google account
   - Contact repository maintainers

2. **Download Problems**:
   - Check internet connection
   - Try different access methods
   - Use download managers for large files

3. **Space Issues**:
   - Use selective sync
   - Clean local cache
   - Archive unused data

## Data Updates

1. **Version Control**:
   - Check last modified dates
   - Review change logs
   - Sync regularly

2. **Contributing**:
   - Document changes
   - Update relevant metadata
   - Notify team members

## Contact

For access requests or technical issues:
- **Author**: Gil Benor
- **Email**: [Your email]
- **Institution**: [Your institution]

## Citation

When using this data, please cite:
```
[Citation information]
``` 