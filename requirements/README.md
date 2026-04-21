# Jupyter Docker Compose

A quick and easy setup for running Jupyter notebooks in a Dockerized environment, managed using [Docker Compose](https://docs.docker.com/compose/). This setup makes it simple to get up and running with Jupyter, share notebooks across multiple team members, and maintain consistent environments. It is also compatible with GitHub Code Spaces for remote development.

## Features

- GitHub Template repository for easy reuse.
- Dockerized Jupyter environment for consistent, reproducible notebook runs.
- Simplified sharing of notebooks using the `work` directory.
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
