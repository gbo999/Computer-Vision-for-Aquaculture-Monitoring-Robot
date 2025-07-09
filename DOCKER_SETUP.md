# Docker Setup for Prawn Research Datasets

This guide explains how to use Docker to work with the prawn research datasets.

## Prerequisites

1. Install Docker and Docker Compose
2. Install Google Drive Desktop App
3. Sync the prawn research data folder

## Quick Start

1. **Set Environment Variables**:
   ```bash
   # Create .env file
   echo "GOOGLE_DRIVE_PATH=~/Google Drive/prawn_research_data" > .env
   ```

2. **Build and Start Services**:
   ```bash
   # Build images
   docker-compose build

   # Start services
   docker-compose up -d
   ```

3. **Access Jupyter Lab**:
   - Open browser at `http://localhost:8888`
   - Notebooks are saved in `./notebooks/`

## Container Services

### 1. Data Processor Container

- Interactive environment for data processing
- Access with:
  ```bash
  docker-compose exec data_processor bash
  ```
- All scripts available in `/app/scripts/`
- Data mounted at `/data/`

### 2. Jupyter Container

- Jupyter Lab interface for interactive analysis
- Ports:
  - 8888: Jupyter Lab
- Data and notebooks persist on host machine

## Directory Structure

```
project/
├── data/                  # Mounted from Google Drive
│   ├── imagej_measurements/
│   ├── molt_exuviae/
│   └── drone_detection/
├── notebooks/            # Jupyter notebooks
├── scripts/              # Processing scripts
├── src/                  # Source code
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Common Tasks

1. **Process Images**:
   ```bash
   # Enter container
   docker-compose exec data_processor bash

   # Run processing script
   python scripts/process_images.py
   ```

2. **Download New Data**:
   ```python
   # In Jupyter notebook
   import gdown

   url = "your_google_drive_url"
   output = "/data/new_folder"
   gdown.download_folder(url, output=output)
   ```

3. **Batch Processing**:
   ```bash
   # Using data_processor service
   docker-compose exec data_processor python scripts/batch_process.py
   ```

## Best Practices

1. **Data Management**:
   - Keep data in Google Drive
   - Use containers for processing only
   - Don't store large files in image

2. **Resource Usage**:
   - Monitor container memory
   - Clean up unused containers
   - Use volume mounts for data

3. **Development**:
   - Test scripts in Jupyter first
   - Move to scripts for production
   - Keep notebooks organized

## Troubleshooting

1. **Mount Issues**:
   - Check Google Drive sync status
   - Verify path in .env file
   - Ensure proper permissions

2. **Container Access**:
   - Check container status:
     ```bash
     docker-compose ps
     ```
   - View logs:
     ```bash
     docker-compose logs
     ```

3. **Performance Issues**:
   - Increase Docker resources
   - Monitor system usage
   - Consider processing in batches

## Updates and Maintenance

1. **Update Images**:
   ```bash
   docker-compose build --no-cache
   ```

2. **Clean Up**:
   ```bash
   # Remove containers
   docker-compose down

   # Clean volumes
   docker-compose down -v
   ```

3. **Backup**:
   - Notebooks saved on host
   - Data backed up via Google Drive
   - Container config in version control 