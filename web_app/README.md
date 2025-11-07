# 🎬 Dubbing-King

A full-stack application for dubbing competitions, featuring a modern Angular frontend and FastAPI backend.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** - Usually included with Docker Desktop

## 🚀 Quick Start

### Step 1: Install Docker

If you haven't already, install Docker on your system:

- **Windows**: Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **macOS**: Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- **Linux**: Follow the [Docker installation guide](https://docs.docker.com/engine/install/)

Verify your installation:
```bash
docker --version
docker compose version
```

### Step 2: Setup Data Directory

Create the required data directory structure for Docker volume mappings:

```bash
# Create data directory
mkdir -p data/uploads

# Create files.db file (empty file, will be populated by the application)
touch data/files.db
```

**Windows PowerShell:**
```powershell
# Create data directory and uploads subfolder
New-Item -ItemType Directory -Force -Path "data\uploads"

# Create files.db file
New-Item -ItemType File -Force -Path "data\files.db"
```

**Windows CMD:**
```cmd
mkdir data\uploads
type nul > data\files.db
```

### Step 3: Build Docker Images

Build the Docker images for both frontend and backend:

```bash
docker compose -f docker-compose.build.yml build
```

This will create two images:
- `dubbing-king-frontend:latest`
- `dubbing-king-backend:latest`

### Step 4: Start the Application

Start all services using Docker Compose:

```bash
docker compose -f docker-compose.run.yml up
```

To run in detached mode (background):
```bash
docker compose -f docker-compose.run.yml up -d
```

## 🌐 Access the Application

Once the containers are running:

- **Frontend**: http://localhost:4200
- **Backend API**: http://localhost:8000

## 🛠️ Useful Commands

### View running containers
```bash
docker compose -f docker-compose.run.yml ps
```

### View logs
```bash
docker compose -f docker-compose.run.yml logs
```

### Stop the application
```bash
docker compose -f docker-compose.run.yml down
```

### Rebuild and restart
```bash
docker compose -f docker-compose.build.yml build
docker compose -f docker-compose.run.yml up -d
```

## 📁 Project Structure

```
web/
├── backend/          # FastAPI backend service
├── dubbing-king/     # Angular frontend application
├── data/            # Data directory (created during setup)
│   ├── uploads/     # Uploaded files storage
│   └── files.db     # Database file
├── docker-compose.build.yml  # Build configuration
└── docker-compose.run.yml    # Runtime configuration
```
