"""
增强版 Web Dashboard - 论文复现展示
===================================

包含:
- 论文 Figure 2 图表展示
- 交互式实验运行
- QUBO 建模可视化
- 完整的论文复现内容
"""

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
import numpy as np
import json
import os
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.algorithms.kt_algorithm import DurhamClustering
from src.algorithms.kmeans_jet import JetKMeans
from src.simulation.event_generator import JetEventGenerator
from src.physics.metrics import compute_event_metrics, MetricsAggregator

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# 论文图表目录
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')


def run_clustering_experiment(n_particles: int, n_jets: int, 
                               algorithm: str, n_events: int = 10) -> dict:
    """运行聚类实验并返回结果"""
    
    generator = JetEventGenerator(
        n_particles=n_particles,
        n_jets=n_jets,
        random_state=42
    )
    
    aggregator = MetricsAggregator()
    sample_event = None
    sample_labels = None
    
    for i in range(n_events):
        event = generator.generate_event()
        
        if algorithm == 'kt':
            clusterer = DurhamClustering()
            result = clusterer.cluster_to_n_jets(event.particles, n_jets)
            labels = result.jet_labels
        elif algorithm == 'kmeans':
            kmeans = JetKMeans(n_clusters=n_jets, n_init=3)
            labels = kmeans.fit_predict(event.particles)
        else:  # QAOA/QUBO - 使用模拟退火
            import kaiwu as kw
            
            x = kw.qubo.ndarray((n_particles, n_jets), 'x', kw.qubo.Binary)
            model = kw.qubo.QuboModel()
            
            D = np.zeros((n_particles, n_particles))
            R = 0.4
            for ii in range(n_particles):
                for jj in range(ii + 1, n_particles):
                    pt_i, eta_i, phi_i = event.particles[ii, :3]
                    pt_j, eta_j, phi_j = event.particles[jj, :3]
                    dphi = phi_i - phi_j
                    while dphi > np.pi: dphi -= 2*np.pi
                    while dphi < -np.pi: dphi += 2*np.pi
                    dR_sq = (eta_i - eta_j)**2 + dphi**2
                    D[ii,jj] = D[jj,ii] = min(pt_i**(-2), pt_j**(-2)) * dR_sq / R**2
            
            obj_terms = []
            for k in range(n_jets):
                for ii in range(n_particles):
                    for jj in range(ii+1, n_particles):
                        if D[ii,jj] > 0:
                            obj_terms.append(D[ii,jj] * x[ii,k] * x[jj,k])
            if obj_terms:
                model.set_objective(kw.qubo.quicksum(obj_terms))
            
            penalty = np.max(D) * n_particles * 10
            for ii in range(n_particles):
                c = kw.qubo.quicksum([x[ii,k] for k in range(n_jets)]) - 1
                model.add_constraint(c == 0, f"p{ii}", penalty=penalty)
            
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=1e5, alpha=0.995,
                cutoff_temperature=0.05, iterations_per_t=50
            )
            controller = kw.common.SolverLoopController(max_repeat_step=5)
            solver = kw.solver.PenaltyMethodSolver(optimizer, controller)
            sol_dict, _ = solver.solve_qubo(model)
            
            labels = np.zeros(n_particles, dtype=int)
            for ii in range(n_particles):
                for k in range(n_jets):
                    if sol_dict.get(f'x[{ii}][{k}]', 0) > 0.5:
                        labels[ii] = k
                        break
        
        metrics = compute_event_metrics(
            event.particles, labels,
            event.true_labels, event.quark_directions
        )
        aggregator.add_event(metrics)
        
        if i == 0:
            sample_event = event
            sample_labels = labels
    
    summary = aggregator.get_summary()
    
    # 生成可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_jets))
    
    for j in range(n_jets):
        mask = sample_labels == j
        if np.any(mask):
            ax.scatter(
                sample_event.particles[mask, 1],
                sample_event.particles[mask, 2],
                c=[colors[j]],
                s=sample_event.particles[mask, 0] * 5,
                label=f'Jet {j+1}',
                alpha=0.7
            )
    
    ax.set_xlabel('η (pseudorapidity)')
    ax.set_ylabel('φ (azimuthal angle)')
    ax.set_title(f'{algorithm.upper()} Clustering Result')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        'avg_angle': round(summary['avg_angle'], 4),
        'angle_error': round(summary['angle_error'], 4),
        'efficiency': round(summary['efficiency_mean'], 4),
        'n_events': n_events,
        'algorithm': algorithm,
        'image': img_base64
    }


@app.route('/')
def index():
    """主仪表板页面"""
    return render_template('index.html')


@app.route('/figures')
def figures():
    """论文 Figure 2 展示页面"""
    return render_template('figures.html')


@app.route('/api/run', methods=['POST'])
def run_experiment():
    """运行聚类实验 API"""
    data = request.get_json()
    
    n_particles = data.get('n_particles', 20)
    n_jets = data.get('n_jets', 2)
    algorithm = data.get('algorithm', 'qaoa')
    n_events = data.get('n_events', 10)
    
    try:
        result = run_clustering_experiment(n_particles, n_jets, algorithm, n_events)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """对比多个算法"""
    data = request.get_json()
    
    n_particles = data.get('n_particles', 20)
    n_jets = data.get('n_jets', 2)
    n_events = data.get('n_events', 10)
    
    results = {}
    for algo in ['qaoa', 'kt', 'kmeans']:
        try:
            results[algo] = run_clustering_experiment(n_particles, n_jets, algo, n_events)
        except Exception as e:
            results[algo] = {'error': str(e)}
    
    return jsonify({'success': True, 'results': results})


@app.route('/api/figure/<figure_name>')
def get_figure(figure_name):
    """获取论文图表"""
    valid_figures = ['fig2a_depth_comparison.png', 'fig2b_k_comparison.png', 
                     'fig2c_algorithm_comparison.png', 'fig2d_quantum_circuit.png',
                     'fig2e_hardware_comparison.png', 'fig2_combined.png']
    
    if figure_name not in valid_figures:
        return jsonify({'error': 'Invalid figure name'}), 404
    
    return send_from_directory(FIGURES_DIR, figure_name)


@app.route('/api/paper_results')
def get_paper_results():
    """获取论文复现结果数据"""
    results_path = os.path.join(FIGURES_DIR, 'experiment_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found'}), 404


@app.route('/api/status')
def get_status():
    """获取系统状态"""
    figures_exist = os.path.exists(os.path.join(FIGURES_DIR, 'fig2_combined.png'))
    return jsonify({
        'status': 'running',
        'version': '2.0.0',
        'algorithms': ['qaoa', 'kt', 'kmeans'],
        'figures_generated': figures_exist
    })


if __name__ == '__main__':
    print("=" * 60)
    print("喷注聚类论文复现 Web Dashboard")
    print("=" * 60)
    print("\n主页: http://localhost:5000")
    print("论文图表: http://localhost:5000/figures")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)
    app.run(debug=True, port=5000)
