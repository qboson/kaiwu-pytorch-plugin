import numpy as np
import csv
from dataclasses import dataclass
from typing import List, Tuple
import os

@dataclass
class Particle:
    pT: float
    eta: float
    phi: float

def _wrap_delta_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def deltaR_matrix(etas: np.ndarray, phis: np.ndarray) -> np.ndarray:
    deta = etas[:, None] - etas[None, :]
    dphi = _wrap_delta_phi(phis[:, None] - phis[None, :])
    return np.sqrt(deta**2 + dphi**2)

def generate_toy_events(
    n_events: int = 100,
    n_particles_range: Tuple[int, int] = (10, 30),
    pT_range: Tuple[float, float] = (1.0, 100.0),
    eta_range: Tuple[float, float] = (-2.5, 2.5),
    seed: int = 42
) -> List[List[Particle]]:
    rng = np.random.default_rng(seed)
    events: List[List[Particle]] = []

    N_min, N_max = n_particles_range
    pT_min, pT_max = pT_range
    eta_min, eta_max = eta_range

    for _ in range(n_events):
        N = int(rng.integers(N_min, N_max + 1))
        pTs = rng.uniform(pT_min, pT_max, size=N)
        etas = rng.uniform(eta_min, eta_max, size=N)
        phis = rng.uniform(0.0, 2.0 * np.pi, size=N)

        event = [Particle(float(pTs[i]), float(etas[i]), float(phis[i])) for i in range(N)]
        events.append(event)

    return events

def event_to_arrays(event: List[Particle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pT = np.array([p.pT for p in event], dtype=float)
    eta = np.array([p.eta for p in event], dtype=float)
    phi = np.array([p.phi for p in event], dtype=float)
    return pT, eta, phi

def save_events_npz(
    events: List[List[Particle]],
    filename: str,
    seed: int,
    n_particles_range: Tuple[int, int],
    pT_range: Tuple[float, float],
    eta_range: Tuple[float, float]
):
    n_events = len(events)
    event_sizes = []
    
    save_dict = {
        "n_events": np.array([n_events], dtype=int),
        "seed": np.array([seed], dtype=int),
        "n_particles_min": np.array([n_particles_range[0]], dtype=int),
        "n_particles_max": np.array([n_particles_range[1]], dtype=int),
        "pT_min": np.array([pT_range[0]], dtype=float),
        "pT_max": np.array([pT_range[1]], dtype=float),
        "eta_min": np.array([eta_range[0]], dtype=float),
        "eta_max": np.array([eta_range[1]], dtype=float)
    }
    
    for idx, event in enumerate(events):
        pT, eta, phi = event_to_arrays(event)
        event_sizes.append(len(event))
        DR = deltaR_matrix(eta, phi)
        save_dict[f"event_{idx}_pT"] = pT
        save_dict[f"event_{idx}_eta"] = eta
        save_dict[f"event_{idx}_phi"] = phi
        save_dict[f"event_{idx}_DR"] = DR
    
    save_dict["event_sizes"] = np.array(event_sizes, dtype=int)
    
    np.savez_compressed(filename, **save_dict)

def save_events_csv(
    events: List[List[Particle]],
    base_filename: str,
    seed: int,
    n_particles_range: Tuple[int, int],
    pT_range: Tuple[float, float],
    eta_range: Tuple[float, float]
):
    os.makedirs(base_filename, exist_ok=True)
    
    params_file = os.path.join(base_filename, "generation_params.csv")
    with open(params_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["seed", seed])
        writer.writerow(["n_particles_min", n_particles_range[0]])
        writer.writerow(["n_particles_max", n_particles_range[1]])
        writer.writerow(["pT_min", pT_range[0]])
        writer.writerow(["pT_max", pT_range[1]])
        writer.writerow(["eta_min", eta_range[0]])
        writer.writerow(["eta_max", eta_range[1]])
        writer.writerow(["n_events", len(events)])
    
    events_table_file = os.path.join(base_filename, "events_table.csv")
    with open(events_table_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "particle_id", "pT", "eta", "phi"])
        for event_id, event in enumerate(events):
            for particle_id, particle in enumerate(event):
                writer.writerow([event_id, particle_id, particle.pT, particle.eta, particle.phi])

if __name__ == "__main__":
    seed = 2025
    n_events = 50
    n_particles_range = (12, 15)
    pT_range = (1.0, 100.0)
    eta_range = (-2.5, 2.5)
    
    events = generate_toy_events(
        n_events=n_events,
        n_particles_range=n_particles_range,
        pT_range=pT_range,
        eta_range=eta_range,
        seed=seed
    )

    if n_events <= 5:
        for idx, ev in enumerate(events):
            pT, eta, phi = event_to_arrays(ev)
            DR = deltaR_matrix(eta, phi)
            print(f"Event {idx}: N={len(ev)}")
            print("pT[:5] =", pT[:5])
            print("eta[:5]=", eta[:5])
            print("phi[:5]=", phi[:5])
            print("ΔR matrix shape:", DR.shape)
            print("ΔR[0,1] =", DR[0, 1])
            print("-" * 50)
    else:
        print(f"Generated {n_events} events (skipping detailed output)")
    
    save_events_npz(
        events,
        "events_data.npz",
        seed,
        n_particles_range,
        pT_range,
        eta_range
    )
    
    save_events_csv(
        events,
        "events_csv",
        seed,
        n_particles_range,
        pT_range,
        eta_range
    )
    
    print(f"\nSaved {n_events} events to:")
    print("  - events_data.npz (NPZ format with ΔR matrices)")
    print("  - events_csv/ (CSV format: events_table.csv + generation_params.csv)")

