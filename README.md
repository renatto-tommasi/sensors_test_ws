# Sensors Test Workspace

This repository contains a Docker setup for developing and testing sensor integrations.

## Prerequisites

- Docker
- Docker Compose

## Getting Started

### Building and Running the Container

1. Clone this repository:
    ```bash
    git clone https://github.com/renatto-tommasi/sensors_test_ws.git
    cd sensors_test_ws
    ```

2. Build and start the Docker container using Docker Compose:
    ```bash
    docker-compose build
    ```

    This command will:
    - Build the Docker image defined in the Dockerfile
    - Start the container with the configured environment
    - Mount the necessary volumes for development

3. To run the container in detached mode (in the background):
    ```bash
    docker-compose up -d
    ```
4. To access the container:
    ```bash
    docker exec -it ros_noetic_container bash
    ```
5. To stop the container:
    ```bash
    docker-compose down
    ```

## Configuration

You can modify the `docker-compose.yml` file to adjust:
- Environmental variables
- Volume mounts
- Port mappings
- Resource limits

## Development

The workspace directory is mounted into the container, allowing you to edit files on your host machine while running code inside the container.

## Troubleshooting

If you encounter any issues with the build process:

1. Check Docker logs:
    ```bash
    docker-compose logs
    ```

2. Ensure all dependencies are correctly specified in the Dockerfile

3. Verify that all required ports are properly exposed in the docker-compose.yml file