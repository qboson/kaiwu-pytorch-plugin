# Jupyter Docker Compose

A quick and easy setup for running Jupyter notebooks in a Dockerized environment, managed using [Docker Compose](https://docs.docker.com/compose/). This setup makes it simple to get up and running with Jupyter, share notebooks across multiple team members, and maintain consistent environments. It is also compatible with GitHub Code Spaces for remote development.

## Features

- Dockerized Jupyter environment for consistent, reproducible notebook runs.
- Save notebooks in `/home/jovyan/work` in Jupyter. This directory is a bind mount of the repository root, so notebook files are saved in the host checkout.
- Compatibility with GitHub Code Spaces for seamless remote development.

## Getting Started

Build the image for the Jupyter Notebook server:

```bash
docker compose build
```

Start the Jupyter Notebook server:

```bash
docker compose up
```

After running this command, the Jupyter Notebook server should be accessible at `http://localhost:8888`.
