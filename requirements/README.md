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


## Network access

Compose publishes host port 8888 on `127.0.0.1` for local development, rather
than on all host interfaces. The notebook still runs without token authentication
and the repository is mounted read-write, so this setup is intended for trusted
local development, not a shared network service. Configure authentication and
review network exposure before enabling remote access.

Keep Docker Engine up to date. See Docker's
[port-publishing documentation](https://docs.docker.com/engine/network/port-publishing/)
for loopback-binding behavior and the limitations of releases older than 28.0.0.
