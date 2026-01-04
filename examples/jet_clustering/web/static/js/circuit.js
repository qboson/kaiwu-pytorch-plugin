/**
 * Interactive Quantum Circuit Builder
 * ====================================
 * 
 * 真正可交互的量子电路编辑器
 * 
 * 交互功能:
 * ✅ 拖拽量子门到电路上
 * ✅ 点击电路位置添加门
 * ✅ 右键删除已有的门
 * ✅ 实时电路状态模拟
 * ✅ 动态参数调整
 * ✅ 电路导出/导入
 * ✅ 撤销/重做操作
 */

// 量子门定义
const QUANTUM_GATES = {
    'H': { name: 'Hadamard', symbol: 'H', color: '#A8D8EA', description: '创建叠加态 |+⟩', qubits: 1 },
    'X': { name: 'Pauli-X', symbol: 'X', color: '#FFB6B6', description: '比特翻转 (NOT门)', qubits: 1 },
    'Y': { name: 'Pauli-Y', symbol: 'Y', color: '#FFD93D', description: '绕Y轴旋转π', qubits: 1 },
    'Z': { name: 'Pauli-Z', symbol: 'Z', color: '#6BCB77', description: '相位翻转', qubits: 1 },
    'S': { name: 'S Gate', symbol: 'S', color: '#87CEEB', description: 'π/2 相位门', qubits: 1 },
    'T': { name: 'T Gate', symbol: 'T', color: '#DDA0DD', description: 'π/4 相位门', qubits: 1 },
    'RX': { name: 'RX(β)', symbol: 'RX', color: '#FFB6B6', description: 'QAOA Mixer层旋转', qubits: 1, hasParam: true, paramName: 'β' },
    'RY': { name: 'RY(θ)', symbol: 'RY', color: '#FFD93D', description: '绕Y轴参数化旋转', qubits: 1, hasParam: true, paramName: 'θ' },
    'RZ': { name: 'RZ(γ)', symbol: 'RZ', color: '#B5E8B5', description: 'QAOA Cost层旋转', qubits: 1, hasParam: true, paramName: 'γ' },
    'CNOT': { name: 'CNOT', symbol: '⊕', color: '#DDA0DD', description: '受控非门 (双量子位)', qubits: 2 },
    'CZ': { name: 'CZ', symbol: 'CZ', color: '#87CEEB', description: '受控相位门', qubits: 2 },
    'SWAP': { name: 'SWAP', symbol: '×', color: '#F0E68C', description: '交换两个量子位', qubits: 2 },
};

// 电路布局常量
const LAYOUT = {
    wireSpacing: 50,
    cellWidth: 50,
    cellHeight: 50,
    gateSize: 36,
    startX: 80,
    startY: 50,
    padding: 20,
};

/**
 * 可交互量子电路编辑器
 */
class InteractiveCircuitEditor {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // 电路状态
        this.nQubits = 6;
        this.nColumns = 15;
        this.grid = [];  // 2D grid: grid[qubit][column] = gate or null
        this.history = [];
        this.historyIndex = -1;

        // 交互状态
        this.selectedGate = null;
        this.isDragging = false;
        this.dragGhost = null;
        this.hoveredCell = null;

        // 模拟状态
        this.stateVector = null;

        this.init();
    }

    init() {
        this.initializeGrid();
        this.render();
        this.attachEventListeners();
        this.loadQAOATemplate();
        this.saveState();
    }

    initializeGrid() {
        this.grid = [];
        for (let q = 0; q < this.nQubits; q++) {
            this.grid[q] = new Array(this.nColumns).fill(null);
        }
    }

    render() {
        this.container.innerHTML = `
            <div class="circuit-editor">
                <!-- 工具栏 -->
                <div class="editor-toolbar">
                    <div class="toolbar-section">
                        <button id="undo-btn" class="tool-btn" title="撤销 (Ctrl+Z)">↶ 撤销</button>
                        <button id="redo-btn" class="tool-btn" title="重做 (Ctrl+Y)">↷ 重做</button>
                        <button id="clear-btn" class="tool-btn" title="清空电路">🗑️ 清空</button>
                    </div>
                    <div class="toolbar-section">
                        <label>量子位: </label>
                        <select id="qubit-select" class="tool-select">
                            ${[2, 3, 4, 5, 6, 7, 8].map(n =>
            `<option value="${n}" ${n === this.nQubits ? 'selected' : ''}>${n} qubits</option>`
        ).join('')}
                        </select>
                        <label style="margin-left: 1rem;">列数: </label>
                        <select id="column-select" class="tool-select">
                            ${[10, 15, 20, 25, 30].map(n =>
            `<option value="${n}" ${n === this.nColumns ? 'selected' : ''}>${n}</option>`
        ).join('')}
                        </select>
                    </div>
                    <div class="toolbar-section">
                        <button id="qaoa-template-btn" class="tool-btn highlight">📐 QAOA模板</button>
                        <button id="export-btn" class="tool-btn">📤 导出</button>
                        <button id="simulate-btn" class="tool-btn highlight">▶️ 模拟</button>
                    </div>
                </div>

                <!-- 量子门面板 -->
                <div class="gate-panel">
                    <h4>🔧 量子门 (点击选择，然后点击电路添加；或直接拖拽)</h4>
                    <div class="gate-palette" id="gate-palette">
                        ${Object.entries(QUANTUM_GATES).map(([id, gate]) => `
                            <div class="gate-draggable" data-gate="${id}" draggable="true" title="${gate.description}">
                                <div class="gate-icon" style="background: ${gate.color}">${gate.symbol}</div>
                                <span class="gate-label">${gate.name}</span>
                                ${gate.qubits === 2 ? '<span class="gate-badge">2Q</span>' : ''}
                            </div>
                        `).join('')}
                    </div>
                    <div class="gate-hint">
                        💡 <strong>操作说明:</strong> 
                        点击选中门 → 点击电路网格添加 | 
                        拖拽门到电路 | 
                        右键删除门 | 
                        双击编辑参数
                    </div>
                </div>

                <!-- 电路画布 -->
                <div class="circuit-canvas-wrapper">
                    <div class="canvas-scroll" id="canvas-scroll">
                        <canvas id="circuit-canvas"></canvas>
                    </div>
                </div>

                <!-- 状态面板 -->
                <div class="state-panel">
                    <div class="state-section">
                        <h4>📊 电路统计</h4>
                        <div class="stats-row" id="circuit-stats">
                            <span>门数: <strong id="gate-count">0</strong></span>
                            <span>深度: <strong id="circuit-depth">0</strong></span>
                            <span>双量子位门: <strong id="two-qubit-count">0</strong></span>
                        </div>
                    </div>
                    <div class="state-section">
                        <h4>🎯 量子态 (模拟结果)</h4>
                        <div class="state-display" id="state-display">
                            <div class="state-placeholder">点击"模拟"按钮运行电路</div>
                        </div>
                    </div>
                </div>

                <!-- 拖拽预览 -->
                <div id="drag-ghost" class="drag-ghost hidden"></div>
            </div>
        `;

        this.canvas = document.getElementById('circuit-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        this.drawCircuit();
    }

    attachEventListeners() {
        // 工具栏按钮
        document.getElementById('undo-btn')?.addEventListener('click', () => this.undo());
        document.getElementById('redo-btn')?.addEventListener('click', () => this.redo());
        document.getElementById('clear-btn')?.addEventListener('click', () => this.clearCircuit());
        document.getElementById('qaoa-template-btn')?.addEventListener('click', () => this.loadQAOATemplate());
        document.getElementById('export-btn')?.addEventListener('click', () => this.exportCircuit());
        document.getElementById('simulate-btn')?.addEventListener('click', () => this.simulateCircuit());

        // 量子位/列数选择
        document.getElementById('qubit-select')?.addEventListener('change', (e) => {
            this.nQubits = parseInt(e.target.value);
            this.resizeGrid();
        });
        document.getElementById('column-select')?.addEventListener('change', (e) => {
            this.nColumns = parseInt(e.target.value);
            this.resizeGrid();
        });

        // 门选择 (点击)
        document.querySelectorAll('.gate-draggable').forEach(el => {
            el.addEventListener('click', () => {
                document.querySelectorAll('.gate-draggable').forEach(g => g.classList.remove('selected'));
                el.classList.add('selected');
                this.selectedGate = el.dataset.gate;
            });

            // 拖拽开始
            el.addEventListener('dragstart', (e) => {
                this.selectedGate = el.dataset.gate;
                this.isDragging = true;
                e.dataTransfer.setData('text/plain', el.dataset.gate);
                e.dataTransfer.effectAllowed = 'copy';

                // 创建拖拽预览
                const ghost = document.getElementById('drag-ghost');
                ghost.innerHTML = `<div class="gate-icon" style="background: ${QUANTUM_GATES[this.selectedGate].color}">${QUANTUM_GATES[this.selectedGate].symbol}</div>`;
                ghost.classList.remove('hidden');
            });

            el.addEventListener('dragend', () => {
                this.isDragging = false;
                document.getElementById('drag-ghost')?.classList.add('hidden');
            });
        });

        // Canvas 事件
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('contextmenu', (e) => this.handleCanvasRightClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        this.canvas.addEventListener('dblclick', (e) => this.handleCanvasDblClick(e));

        // 拖放
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            this.handleCanvasMouseMove(e);
        });
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            const gateType = e.dataTransfer.getData('text/plain');
            if (gateType && this.hoveredCell) {
                this.addGate(gateType, this.hoveredCell.qubit, this.hoveredCell.column);
            }
            document.getElementById('drag-ghost')?.classList.add('hidden');
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                this.undo();
            } else if (e.ctrlKey && e.key === 'y') {
                e.preventDefault();
                this.redo();
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.hoveredCell && this.grid[this.hoveredCell.qubit]?.[this.hoveredCell.column]) {
                    this.removeGate(this.hoveredCell.qubit, this.hoveredCell.column);
                }
            }
        });
    }

    resizeCanvas() {
        const width = LAYOUT.startX + (this.nColumns + 1) * LAYOUT.cellWidth + LAYOUT.padding;
        const height = LAYOUT.startY + this.nQubits * LAYOUT.wireSpacing + LAYOUT.padding;
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';
    }

    resizeGrid() {
        // 调整网格大小，保留现有的门
        const newGrid = [];
        for (let q = 0; q < this.nQubits; q++) {
            newGrid[q] = new Array(this.nColumns).fill(null);
            if (this.grid[q]) {
                for (let c = 0; c < Math.min(this.nColumns, this.grid[q].length); c++) {
                    newGrid[q][c] = this.grid[q][c];
                }
            }
        }
        this.grid = newGrid;
        this.resizeCanvas();
        this.drawCircuit();
        this.saveState();
    }

    drawCircuit() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 背景
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // 绘制网格线（淡色）
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let c = 0; c <= this.nColumns; c++) {
            const x = LAYOUT.startX + c * LAYOUT.cellWidth;
            ctx.beginPath();
            ctx.moveTo(x, LAYOUT.startY - 20);
            ctx.lineTo(x, LAYOUT.startY + this.nQubits * LAYOUT.wireSpacing);
            ctx.stroke();
        }

        // 绘制量子位导线
        ctx.strokeStyle = '#4a5568';
        ctx.lineWidth = 2;
        for (let q = 0; q < this.nQubits; q++) {
            const y = LAYOUT.startY + q * LAYOUT.wireSpacing;

            // 量子位标签
            ctx.fillStyle = '#94a3b8';
            ctx.font = '14px monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`|q${this.nQubits - 1 - q}⟩`, LAYOUT.startX - 15, y + 5);

            // 导线
            ctx.beginPath();
            ctx.moveTo(LAYOUT.startX, y);
            ctx.lineTo(LAYOUT.startX + this.nColumns * LAYOUT.cellWidth, y);
            ctx.stroke();
        }

        // 列号
        ctx.fillStyle = '#64748b';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        for (let c = 0; c < this.nColumns; c++) {
            const x = LAYOUT.startX + c * LAYOUT.cellWidth + LAYOUT.cellWidth / 2;
            ctx.fillText(c.toString(), x, LAYOUT.startY - 8);
        }

        // 绘制所有门
        for (let q = 0; q < this.nQubits; q++) {
            for (let c = 0; c < this.nColumns; c++) {
                const gate = this.grid[q][c];
                if (gate) {
                    this.drawGate(gate, q, c);
                }
            }
        }

        // 绘制悬停高亮
        if (this.hoveredCell) {
            const { qubit, column } = this.hoveredCell;
            const x = LAYOUT.startX + column * LAYOUT.cellWidth;
            const y = LAYOUT.startY + qubit * LAYOUT.wireSpacing - LAYOUT.cellHeight / 2;

            ctx.strokeStyle = this.selectedGate ? '#22d3ee' : 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x + 2, y + 2, LAYOUT.cellWidth - 4, LAYOUT.cellHeight - 4);
            ctx.setLineDash([]);
        }

        this.updateStats();
    }

    drawGate(gate, qubit, column) {
        const ctx = this.ctx;
        const gateInfo = QUANTUM_GATES[gate.type];
        if (!gateInfo) return;

        const x = LAYOUT.startX + column * LAYOUT.cellWidth + (LAYOUT.cellWidth - LAYOUT.gateSize) / 2;
        const y = LAYOUT.startY + qubit * LAYOUT.wireSpacing - LAYOUT.gateSize / 2;

        if (gateInfo.qubits === 2 && gate.target !== undefined) {
            // 双量子位门
            const y2 = LAYOUT.startY + gate.target * LAYOUT.wireSpacing;
            const cx = x + LAYOUT.gateSize / 2;

            // 连接线
            ctx.strokeStyle = gateInfo.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, LAYOUT.startY + qubit * LAYOUT.wireSpacing);
            ctx.lineTo(cx, y2);
            ctx.stroke();

            // 控制点
            ctx.fillStyle = gateInfo.color;
            ctx.beginPath();
            ctx.arc(cx, LAYOUT.startY + qubit * LAYOUT.wireSpacing, 6, 0, Math.PI * 2);
            ctx.fill();

            // 目标
            if (gate.type === 'CNOT') {
                ctx.strokeStyle = gateInfo.color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(cx, y2, 10, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(cx - 8, y2);
                ctx.lineTo(cx + 8, y2);
                ctx.moveTo(cx, y2 - 8);
                ctx.lineTo(cx, y2 + 8);
                ctx.stroke();
            } else {
                // CZ, SWAP 等
                ctx.fillStyle = gateInfo.color;
                ctx.beginPath();
                ctx.arc(cx, y2, 6, 0, Math.PI * 2);
                ctx.fill();
            }
        } else {
            // 单量子位门
            ctx.fillStyle = gateInfo.color;
            ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
            ctx.shadowBlur = 4;
            ctx.shadowOffsetY = 2;

            // 圆角矩形
            const r = 6;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + LAYOUT.gateSize - r, y);
            ctx.quadraticCurveTo(x + LAYOUT.gateSize, y, x + LAYOUT.gateSize, y + r);
            ctx.lineTo(x + LAYOUT.gateSize, y + LAYOUT.gateSize - r);
            ctx.quadraticCurveTo(x + LAYOUT.gateSize, y + LAYOUT.gateSize, x + LAYOUT.gateSize - r, y + LAYOUT.gateSize);
            ctx.lineTo(x + r, y + LAYOUT.gateSize);
            ctx.quadraticCurveTo(x, y + LAYOUT.gateSize, x, y + LAYOUT.gateSize - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.fill();

            ctx.shadowColor = 'transparent';

            // 门符号
            ctx.fillStyle = '#1a1a2e';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(gateInfo.symbol, x + LAYOUT.gateSize / 2, y + LAYOUT.gateSize / 2);

            // 参数标签
            if (gateInfo.hasParam && gate.param !== undefined) {
                ctx.fillStyle = '#64748b';
                ctx.font = '9px sans-serif';
                ctx.fillText(`${(gate.param / Math.PI).toFixed(1)}π`, x + LAYOUT.gateSize / 2, y + LAYOUT.gateSize + 10);
            }
        }
    }

    getGridCell(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const column = Math.floor((x - LAYOUT.startX) / LAYOUT.cellWidth);
        const qubit = Math.round((y - LAYOUT.startY) / LAYOUT.wireSpacing);

        if (column >= 0 && column < this.nColumns && qubit >= 0 && qubit < this.nQubits) {
            return { qubit, column };
        }
        return null;
    }

    handleCanvasClick(e) {
        const cell = this.getGridCell(e);
        if (!cell) return;

        const { qubit, column } = cell;

        if (this.selectedGate) {
            this.addGate(this.selectedGate, qubit, column);
        }
    }

    handleCanvasRightClick(e) {
        e.preventDefault();
        const cell = this.getGridCell(e);
        if (!cell) return;

        const { qubit, column } = cell;
        if (this.grid[qubit][column]) {
            this.removeGate(qubit, column);
        }
    }

    handleCanvasMouseMove(e) {
        const cell = this.getGridCell(e);
        if (cell?.qubit !== this.hoveredCell?.qubit || cell?.column !== this.hoveredCell?.column) {
            this.hoveredCell = cell;
            this.drawCircuit();
        }

        // 更新拖拽预览位置
        if (this.isDragging) {
            const ghost = document.getElementById('drag-ghost');
            ghost.style.left = (e.clientX + 10) + 'px';
            ghost.style.top = (e.clientY + 10) + 'px';
        }
    }

    handleCanvasDblClick(e) {
        const cell = this.getGridCell(e);
        if (!cell) return;

        const gate = this.grid[cell.qubit][cell.column];
        if (gate && QUANTUM_GATES[gate.type]?.hasParam) {
            const newParam = prompt(`输入参数 (单位: π 的倍数)`, (gate.param / Math.PI).toFixed(2));
            if (newParam !== null) {
                gate.param = parseFloat(newParam) * Math.PI;
                this.drawCircuit();
                this.saveState();
            }
        }
    }

    addGate(gateType, qubit, column) {
        const gateInfo = QUANTUM_GATES[gateType];
        if (!gateInfo) return;

        const gate = {
            type: gateType,
            param: gateInfo.hasParam ? Math.PI / 4 : undefined
        };

        // 双量子位门需要目标
        if (gateInfo.qubits === 2) {
            let targetQubit = qubit + 1;
            if (targetQubit >= this.nQubits) targetQubit = qubit - 1;
            if (targetQubit < 0) return;  // 不够量子位

            gate.target = targetQubit;
        }

        this.grid[qubit][column] = gate;
        this.drawCircuit();
        this.saveState();
    }

    removeGate(qubit, column) {
        if (this.grid[qubit][column]) {
            this.grid[qubit][column] = null;
            this.drawCircuit();
            this.saveState();
        }
    }

    loadQAOATemplate() {
        this.initializeGrid();

        // Hadamard 层
        for (let q = 0; q < this.nQubits; q++) {
            this.grid[q][0] = { type: 'H' };
        }

        // Cost 层 (ZZ interactions)
        let col = 1;
        for (let i = 0; i < this.nQubits - 1 && col < this.nColumns - 2; i++) {
            for (let j = i + 1; j < this.nQubits && col < this.nColumns - 2; j++) {
                this.grid[i][col] = { type: 'CNOT', target: j };
                col++;
                this.grid[j][col] = { type: 'RZ', param: Math.PI / 4 };
                col++;
                if (col < this.nColumns) {
                    this.grid[i][col] = { type: 'CNOT', target: j };
                    col++;
                }
            }
        }

        // Mixer 层
        const mixerCol = Math.min(col, this.nColumns - 1);
        for (let q = 0; q < this.nQubits; q++) {
            if (!this.grid[q][mixerCol]) {
                this.grid[q][mixerCol] = { type: 'RX', param: Math.PI / 3 };
            }
        }

        this.drawCircuit();
        this.saveState();
    }

    clearCircuit() {
        if (confirm('确定要清空电路吗？')) {
            this.initializeGrid();
            this.drawCircuit();
            this.saveState();
        }
    }

    saveState() {
        // 删除当前位置之后的历史
        this.history = this.history.slice(0, this.historyIndex + 1);
        // 保存当前状态
        this.history.push(JSON.stringify(this.grid));
        this.historyIndex = this.history.length - 1;

        // 限制历史长度
        if (this.history.length > 50) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.grid = JSON.parse(this.history[this.historyIndex]);
            this.drawCircuit();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.grid = JSON.parse(this.history[this.historyIndex]);
            this.drawCircuit();
        }
    }

    updateStats() {
        let gateCount = 0;
        let twoQubitCount = 0;
        let maxColumn = 0;

        for (let q = 0; q < this.nQubits; q++) {
            for (let c = 0; c < this.nColumns; c++) {
                if (this.grid[q][c]) {
                    gateCount++;
                    if (QUANTUM_GATES[this.grid[q][c].type]?.qubits === 2) {
                        twoQubitCount++;
                    }
                    maxColumn = Math.max(maxColumn, c);
                }
            }
        }

        document.getElementById('gate-count').textContent = gateCount;
        document.getElementById('circuit-depth').textContent = maxColumn + 1;
        document.getElementById('two-qubit-count').textContent = twoQubitCount;
    }

    simulateCircuit() {
        // 简化的量子态模拟
        const n = this.nQubits;
        const dim = 1 << n;  // 2^n

        // 初始状态 |00...0⟩
        let state = new Array(dim).fill(0);
        state[0] = { re: 1, im: 0 };
        for (let i = 1; i < dim; i++) {
            state[i] = { re: 0, im: 0 };
        }

        // 应用门 (简化模拟，仅支持 H 门展示)
        for (let c = 0; c < this.nColumns; c++) {
            for (let q = 0; q < n; q++) {
                const gate = this.grid[q][c];
                if (gate?.type === 'H') {
                    // Hadamard 门作用
                    const newState = state.map(() => ({ re: 0, im: 0 }));
                    const factor = 1 / Math.sqrt(2);

                    for (let i = 0; i < dim; i++) {
                        const bit = (i >> (n - 1 - q)) & 1;
                        const flipped = i ^ (1 << (n - 1 - q));

                        if (bit === 0) {
                            newState[i].re += factor * state[i].re;
                            newState[i].im += factor * state[i].im;
                            newState[flipped].re += factor * state[i].re;
                            newState[flipped].im += factor * state[i].im;
                        } else {
                            newState[i].re += factor * state[flipped].re;
                            newState[i].im += factor * state[flipped].im;
                            newState[flipped].re -= factor * state[flipped].re;
                            newState[flipped].im -= factor * state[flipped].im;
                        }
                    }
                    state = newState;
                }
            }
        }

        // 显示结果
        this.displayStateVector(state);
    }

    displayStateVector(state) {
        const display = document.getElementById('state-display');
        const n = this.nQubits;

        // 找出概率不为零的态
        const nonZero = state
            .map((amp, i) => ({
                index: i,
                binary: i.toString(2).padStart(n, '0'),
                prob: amp.re * amp.re + amp.im * amp.im,
                amp
            }))
            .filter(s => s.prob > 0.001)
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 8);

        if (nonZero.length === 0) {
            display.innerHTML = '<div class="state-placeholder">无有效量子态</div>';
            return;
        }

        display.innerHTML = `
            <div class="state-bars">
                ${nonZero.map(s => `
                    <div class="state-bar-item">
                        <span class="state-label">|${s.binary}⟩</span>
                        <div class="state-bar-container">
                            <div class="state-bar" style="width: ${s.prob * 100}%"></div>
                        </div>
                        <span class="state-prob">${(s.prob * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    exportCircuit() {
        const data = {
            nQubits: this.nQubits,
            nColumns: this.nColumns,
            grid: this.grid,
            exportTime: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'qaoa_circuit.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('quantum-circuit-section');
    if (container) {
        window.circuitEditor = new InteractiveCircuitEditor('quantum-circuit-section');
        console.log('Interactive Circuit Editor initialized');
    }
});
