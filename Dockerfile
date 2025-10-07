# 1. Select base image for building the container
FROM python:3.10-slim

# 2. Set environment variable to ensure Python output is visible in Docker logs
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory for subsequent commands
WORKDIR /app

# 4. Upgrade pip
RUN python -m pip install --upgrade pip

# 5. Copy only the requirements file to the container
COPY requirements.txt /app/

# 6. Install dependencies specified in the requirements file
RUN python -m pip install --no-cache-dir -r requirements.txt

# 7. Copy the src folder to the app directory inside the container and the rest of the application code into the container
COPY src /app/src
COPY . /app/

# 8. Copy Docker-specific environment file
COPY .env.docker /app/.env

# 9. Set environment variable to include src in PYTHONPATH
ENV PYTHONPATH=/app/src

# 10. Copy and make entrypoint script executable
COPY entrypoint-wrapper.sh /app/entrypoint-wrapper.sh
RUN chmod +x /app/entrypoint-wrapper.sh

# 11. Set the entry point to the script
ENTRYPOINT ["/app/entrypoint-wrapper.sh"]
