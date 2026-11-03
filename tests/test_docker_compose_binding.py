"""The local notebook example must not publish its port on every host interface."""

from ipaddress import ip_address
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def test_notebook_port_is_explicitly_loopback_bound():
    """Parse Compose's short port syntax and retain the documented notebook port."""
    path = Path(__file__).resolve().parents[1] / "requirements" / "docker-compose.yml"
    service = yaml.safe_load(path.read_text(encoding="utf-8"))["services"]["jupyter"]
    assert len(service["ports"]) == 1
    mapping = service["ports"][0]
    assert isinstance(mapping, str)
    parts = mapping.rsplit(":", 2)
    assert len(parts) == 3, "The notebook port needs an explicit loopback host address"
    host_ip, host_port, container_port = parts
    assert ip_address(host_ip.strip("[]")).is_loopback
    assert host_port == container_port == "8888"
