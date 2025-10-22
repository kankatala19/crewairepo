class CrewAIManager {
    constructor() {
        this.promptInput = document.getElementById('promptInput');
        this.startCrewButton = document.getElementById('startCrewButton');
        this.attachFilesButton = document.getElementById('attachFilesButton');
        this.fileInput = document.getElementById('fileInput');
        this.selectedFiles = document.getElementById('selectedFiles');
        this.charCount = document.getElementById('charCount');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.outputSection = document.getElementById('outputSection');
        this.outputContent = document.getElementById('outputContent');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        
        // Agent elements
        this.plannerCard = document.getElementById('plannerCard');
        this.writerCard = document.getElementById('writerCard');
        this.reviewerCard = document.getElementById('reviewerCard');
        this.plannerStatus = document.getElementById('plannerStatus');
        this.writerStatus = document.getElementById('writerStatus');
        this.reviewerStatus = document.getElementById('reviewerStatus');
        
        // Action buttons
        this.copyOutputButton = document.getElementById('copyOutputButton');
        this.downloadOutputButton = document.getElementById('downloadOutputButton');
        this.resetButton = document.getElementById('resetButton');
        
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        this.menuToggle = document.getElementById('menuToggle');
        this.historyToggle = document.getElementById('historyToggle');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.newChatButton = document.getElementById('newChatButton');
        this.mainContainer = document.querySelector('.main-container');
        
        // History elements
        this.sidebarHistoryList = document.getElementById('sidebarHistoryList');
        this.sidebarHistoryLoading = document.getElementById('sidebarHistoryLoading');
        this.sidebarHistoryEmpty = document.getElementById('sidebarHistoryEmpty');
        this.clearHistoryButton = document.getElementById('clearHistoryButton');
        
        // Modals
        this.errorModal = document.getElementById('errorModal');
        this.successModal = document.getElementById('successModal');
        this.taskDetailsModal = document.getElementById('taskDetailsModal');
        this.confirmModal = document.getElementById('confirmModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.successMessage = document.getElementById('successMessage');
        this.confirmMessage = document.getElementById('confirmMessage');
        
        // Task details modal elements
        this.taskStatus = document.getElementById('taskStatus');
        this.taskCreated = document.getElementById('taskCreated');
        this.taskCompleted = document.getElementById('taskCompleted');
        this.taskPromptContent = document.getElementById('taskPromptContent');
        this.taskOutputContent = document.getElementById('taskOutputContent');
        
        this.statusCheckInterval = null;
        this.isRunning = false;
        this.currentTaskId = null;
        this.pendingAction = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.autoResizeTextarea();
        this.focusInput();
        this.updateStatus('ready', 'Ready');
        this.loadHistory();
    }
    
    setupEventListeners() {
        // Start crew button
        this.startCrewButton.addEventListener('click', () => this.startCrew());
        
        // Keyboard shortcuts
        this.promptInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.startCrew();
            }
        });
        
        // Character count update
        this.promptInput.addEventListener('input', () => this.updateCharCount());
        
        // Action buttons
        this.copyOutputButton.addEventListener('click', () => this.copyOutput());
        this.downloadOutputButton.addEventListener('click', () => this.downloadOutput());
        this.resetButton.addEventListener('click', () => this.resetCrew());
        this.attachFilesButton.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', () => this.updateSelectedFiles());
        
        // Sidebar buttons
        this.menuToggle.addEventListener('click', () => this.toggleSidebar());
        this.historyToggle.addEventListener('click', () => this.toggleHistorySidebar());
        this.sidebarToggle.addEventListener('click', () => this.closeSidebar());
        this.newChatButton.addEventListener('click', () => this.startNewTask());
        this.clearHistoryButton.addEventListener('click', () => this.confirmClearHistory());
        
        // Modal close events
        document.getElementById('closeModal').addEventListener('click', () => this.hideErrorModal());
        document.getElementById('okButton').addEventListener('click', () => this.hideErrorModal());
        document.getElementById('closeSuccessModal').addEventListener('click', () => this.hideSuccessModal());
        document.getElementById('okSuccessButton').addEventListener('click', () => this.hideSuccessModal());
        document.getElementById('closeTaskDetailsModal').addEventListener('click', () => this.hideTaskDetailsModal());
        document.getElementById('closeTaskDetailsButton').addEventListener('click', () => this.hideTaskDetailsModal());
        document.getElementById('closeConfirmModal').addEventListener('click', () => this.hideConfirmModal());
        document.getElementById('cancelConfirmButton').addEventListener('click', () => this.hideConfirmModal());
        document.getElementById('confirmButton').addEventListener('click', () => this.executePendingAction());
        document.getElementById('deleteTaskButton').addEventListener('click', () => this.confirmDeleteTask());
        
        // Close modals on backdrop click
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) this.hideErrorModal();
        });
        this.successModal.addEventListener('click', (e) => {
            if (e.target === this.successModal) this.hideSuccessModal();
        });
        this.taskDetailsModal.addEventListener('click', (e) => {
            if (e.target === this.taskDetailsModal) this.hideTaskDetailsModal();
        });
        this.confirmModal.addEventListener('click', (e) => {
            if (e.target === this.confirmModal) this.hideConfirmModal();
        });
        
        // Close modals on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideErrorModal();
                this.hideSuccessModal();
                this.hideTaskDetailsModal();
                this.hideConfirmModal();
            }
        });
    }
    
    autoResizeTextarea() {
        this.promptInput.addEventListener('input', () => {
            this.promptInput.style.height = 'auto';
            this.promptInput.style.height = Math.min(this.promptInput.scrollHeight, 200) + 'px';
        });
    }
    
    updateCharCount() {
        if (!this.charCount) return;
        const count = this.promptInput.value.length;
        this.charCount.textContent = `${count}`;
        this.charCount.style.color = '#64748b';
    }
    
    focusInput() {
        this.promptInput.focus();
    }
    
    async startCrew() {
        const prompt = this.promptInput.value.trim();
        
        if (!prompt) {
            this.showError('Please enter a task or query');
            return;
        }
        
        if (prompt.length > 2000) {
            this.showError('Prompt is too long. Please keep it under 2000 characters.');
            return;
        }
        
        if (this.isRunning) {
            this.showError('Crew is already running. Please wait for completion.');
            return;
        }
        
        try {
            this.isRunning = true;
            this.updateStatus('running', 'Running');
            this.resetAgentStates();
            this.startCrewButton.disabled = true;
            this.promptInput.disabled = true;
            this.showLoadingOverlay('Starting crew...');
            
            const formData = new FormData();
            formData.append('prompt', prompt);
            if (this.fileInput && this.fileInput.files && this.fileInput.files.length > 0) {
                Array.from(this.fileInput.files).forEach(file => formData.append('files', file));
            }
            const response = await fetch('/start_crew', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to start crew');
            }
            
            const data = await response.json();
            this.currentTaskId = data.task_id;
            
            // Clear the prompt input and selected files after starting the crew
            this.promptInput.value = '';
            this.promptInput.style.height = 'auto';
            this.updateCharCount();
            if (this.fileInput) {
                this.fileInput.value = '';
            }
            if (this.selectedFiles) {
                this.selectedFiles.style.display = 'none';
                this.selectedFiles.innerHTML = '';
            }
            
            this.hideLoadingOverlay();
            this.startStatusPolling();
            
        } catch (error) {
            this.isRunning = false;
            this.updateStatus('error', 'Error');
            this.startCrewButton.disabled = false;
            this.promptInput.disabled = false;
            this.hideLoadingOverlay();
            this.showError(error.message);
        }
    }

    updateSelectedFiles() {
        if (!this.selectedFiles) return;
        const files = Array.from(this.fileInput.files || []);
        if (files.length === 0) {
            this.selectedFiles.style.display = 'none';
            this.selectedFiles.innerHTML = '';
            return;
        }
        this.selectedFiles.style.display = 'block';
        this.selectedFiles.innerHTML = files.map(f => `<div class="file-pill" title="${f.name}"><i class="fas fa-file"></i> ${this.truncateText(f.name, 24)}</div>`).join('');
    }
    
    startStatusPolling() {
        this.statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/crew_status');
                const status = await response.json();
                
                this.updateProgress(status.progress);
                this.updateAgentStates(status);
                
                if (!status.is_running) {
                    this.stopStatusPolling();
                    
                    if (status.error) {
                        this.updateStatus('error', 'Error');
                        this.showError(status.error);
                    } else {
                        this.updateStatus('complete', 'Complete');
                        await this.loadFinalOutput();
                    }
                    
                    // Refresh history after task completion
                    this.loadHistory();
                    
                    this.startCrewButton.disabled = false;
                    this.promptInput.disabled = false;
                    this.isRunning = false;
                }
                
            } catch (error) {
                console.error('Error checking status:', error);
                this.stopStatusPolling();
                this.updateStatus('error', 'Error');
                this.showError('Failed to check crew status');
                this.startCrewButton.disabled = false;
                this.promptInput.disabled = false;
                this.isRunning = false;
            }
        }, 1000);
    }
    
    stopStatusPolling() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }
    
    updateProgress(progress) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${progress}%`;
    }
    
    updateAgentStates(status) {
        const currentAgent = status.current_agent;
        
        // Reset all agents
        this.resetAgentStates();
        
        if (currentAgent === 'Planner') {
            this.setAgentActive('planner', status.current_task);
        } else if (currentAgent === 'Writer') {
            this.setAgentCompleted('planner');
            this.setAgentActive('writer', status.current_task);
        } else if (currentAgent === 'Reviewer') {
            this.setAgentCompleted('planner');
            this.setAgentCompleted('writer');
            this.setAgentActive('reviewer', status.current_task);
        } else if (currentAgent === 'Complete') {
            this.setAgentCompleted('planner');
            this.setAgentCompleted('writer');
            this.setAgentCompleted('reviewer');
        } else if (currentAgent === 'Error') {
            this.setAgentError('planner');
            this.setAgentError('writer');
            this.setAgentError('reviewer');
        }
    }
    
    resetAgentStates() {
        this.setAgentWaiting('planner');
        this.setAgentWaiting('writer');
        this.setAgentWaiting('reviewer');
    }
    
    setAgentWaiting(agent) {
        const card = document.getElementById(`${agent}Card`);
        const status = document.getElementById(`${agent}Status`);
        
        card.className = 'agent-card waiting';
        status.textContent = 'Waiting';
    }
    
    setAgentActive(agent, task) {
        const card = document.getElementById(`${agent}Card`);
        const status = document.getElementById(`${agent}Status`);
        
        card.className = 'agent-card active';
        status.textContent = task || 'Working...';
    }
    
    setAgentCompleted(agent) {
        const card = document.getElementById(`${agent}Card`);
        const status = document.getElementById(`${agent}Status`);
        
        card.className = 'agent-card completed';
        status.textContent = 'Completed';
    }
    
    setAgentError(agent) {
        const card = document.getElementById(`${agent}Card`);
        const status = document.getElementById(`${agent}Status`);
        
        card.className = 'agent-card error';
        status.textContent = 'Error';
    }
    
    async loadFinalOutput() {
        try {
            const response = await fetch('/crew_output');
            const data = await response.json();
            
            if (data.is_complete && data.output) {
                this.outputContent.textContent = data.output;
                this.outputSection.style.display = 'block';
                this.outputSection.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (error) {
            console.error('Error loading output:', error);
            this.showError('Failed to load final output');
        }
    }
    
    updateStatus(type, text) {
        this.statusDot.className = `status-dot ${type}`;
        this.statusText.textContent = text;
    }
    
    showLoadingOverlay(text) {
        this.loadingText.textContent = text;
        this.loadingOverlay.classList.add('show');
    }
    
    hideLoadingOverlay() {
        this.loadingOverlay.classList.remove('show');
    }
    
    async copyOutput() {
        try {
            await navigator.clipboard.writeText(this.outputContent.textContent);
            this.showSuccess('Output copied to clipboard!');
        } catch (error) {
            console.error('Failed to copy:', error);
            this.showError('Failed to copy output to clipboard');
        }
    }
    
    downloadOutput() {
        const content = this.outputContent.textContent;
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `crewai-output-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showSuccess('Output downloaded successfully!');
    }
    
    async resetCrew() {
        try {
            const response = await fetch('/reset_crew', { method: 'POST' });
            if (response.ok) {
                this.resetAgentStates();
                this.updateProgress(0);
                this.updateStatus('ready', 'Ready');
                this.outputSection.style.display = 'none';
                this.outputContent.textContent = '';
                this.promptInput.value = '';
                this.promptInput.style.height = 'auto';
                this.updateCharCount();
                this.focusInput();
                this.showSuccess('Crew reset successfully!');
            }
        } catch (error) {
            console.error('Error resetting crew:', error);
            this.showError('Failed to reset crew');
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.add('show');
    }
    
    hideErrorModal() {
        this.errorModal.classList.remove('show');
    }
    
    showSuccess(message) {
        this.successMessage.textContent = message;
        this.successModal.classList.add('show');
    }
    
    hideSuccessModal() {
        this.successModal.classList.remove('show');
    }
    
    // Sidebar Methods
    toggleSidebar() {
        this.sidebar.classList.toggle('closed');
        this.mainContainer.classList.toggle('sidebar-closed');
        
        // Load history when sidebar opens
        if (!this.sidebar.classList.contains('closed')) {
            this.loadHistory();
        }
    }
    
    toggleHistorySidebar() {
        this.sidebar.classList.toggle('closed');
        this.mainContainer.classList.toggle('sidebar-closed');
        this.historyToggle.classList.toggle('active');
        
        // Load history when sidebar opens
        if (!this.sidebar.classList.contains('closed')) {
            this.loadHistory();
        }
    }
    
    closeSidebar() {
        this.sidebar.classList.add('closed');
        this.mainContainer.classList.add('sidebar-closed');
        this.historyToggle.classList.remove('active');
    }
    
    openSidebar() {
        this.sidebar.classList.remove('closed');
        this.mainContainer.classList.remove('sidebar-closed');
        this.historyToggle.classList.add('active');
        this.loadHistory();
    }
    
    startNewTask() {
        this.closeSidebar();
        this.resetCrew();
        this.promptInput.focus();
    }
    
    // History Methods
    async loadHistory() {
        try {
            this.showSidebarHistoryLoading();
            
            const response = await fetch('/history');
            if (!response.ok) {
                throw new Error('Failed to load history');
            }
            
            const data = await response.json();
            this.displaySidebarHistory(data.tasks);
            
        } catch (error) {
            console.error('Error loading history:', error);
            this.showError('Failed to load history');
            this.showSidebarHistoryEmpty();
        }
    }
    
    showSidebarHistoryLoading() {
        this.sidebarHistoryLoading.style.display = 'flex';
        this.sidebarHistoryList.style.display = 'none';
        this.sidebarHistoryEmpty.style.display = 'none';
    }
    
    showSidebarHistoryEmpty() {
        this.sidebarHistoryLoading.style.display = 'none';
        this.sidebarHistoryList.style.display = 'none';
        this.sidebarHistoryEmpty.style.display = 'flex';
    }
    
    displaySidebarHistory(tasks) {
        this.sidebarHistoryLoading.style.display = 'none';
        this.sidebarHistoryEmpty.style.display = 'none';
        this.sidebarHistoryList.style.display = 'block';
        
        if (tasks.length === 0) {
            this.showSidebarHistoryEmpty();
            return;
        }
        
        this.sidebarHistoryList.innerHTML = tasks.map(task => this.createSidebarHistoryItem(task)).join('');
        
        // Add event listeners to history items
        this.sidebarHistoryList.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-action')) {
                    this.loadTaskFromHistory(item.dataset.taskId);
                }
            });
        });
        
        this.sidebarHistoryList.querySelectorAll('.history-item-action').forEach(action => {
            action.addEventListener('click', (e) => {
                e.stopPropagation();
                const taskId = action.closest('.history-item').dataset.taskId;
                const actionType = action.dataset.action;
                
                if (actionType === 'view') {
                    this.showTaskDetails(taskId);
                } else if (actionType === 'delete') {
                    this.confirmDeleteTask(taskId);
                }
            });
        });
    }
    
    createSidebarHistoryItem(task) {
        return `
            <div class="history-item ${task.status}" data-task-id="${task.id}">
                <div class="history-item-icon">
                    <i class="fas fa-${task.status === 'completed' ? 'check' : task.status === 'running' ? 'clock' : 'exclamation'}"></i>
                </div>
                <div class="history-item-content">
                    <div class="history-item-title">${this.truncateText(task.prompt, 50)}</div>
                </div>
                <div class="history-item-actions">
                    <button class="history-item-action" data-action="view" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="history-item-action delete" data-action="delete" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }
    
    async loadTaskFromHistory(taskId) {
        try {
            const response = await fetch(`/history/${taskId}`);
            if (!response.ok) {
                throw new Error('Failed to load task');
            }
            
            const task = await response.json();
            
            // Load the task into the current interface
            this.promptInput.value = task.prompt;
            this.promptInput.style.height = 'auto';
            this.updateCharCount();
            
            if (task.status === 'completed' && task.output) {
                this.outputContent.textContent = task.output;
                this.outputSection.style.display = 'block';
            }
            
            this.closeSidebar();
            this.promptInput.focus();
            
        } catch (error) {
            console.error('Error loading task:', error);
            this.showError('Failed to load task from history');
        }
    }
    
    async showTaskDetails(taskId) {
        try {
            const response = await fetch(`/history/${taskId}`);
            if (!response.ok) {
                throw new Error('Failed to load task details');
            }
            
            const task = await response.json();
            
            // Populate task details modal
            this.taskStatus.textContent = task.status;
            this.taskStatus.className = `task-status ${task.status}`;
            this.taskCreated.textContent = new Date(task.created_at).toLocaleString();
            this.taskCompleted.textContent = task.completed_at ? 
                new Date(task.completed_at).toLocaleString() : 'Not completed';
            this.taskPromptContent.textContent = task.prompt;
            this.taskOutputContent.textContent = task.output || 'No output available';
            
            // Store current task ID for deletion
            this.currentTaskId = taskId;
            
            this.taskDetailsModal.classList.add('show');
            
        } catch (error) {
            console.error('Error loading task details:', error);
            this.showError('Failed to load task details');
        }
    }
    
    hideTaskDetailsModal() {
        this.taskDetailsModal.classList.remove('show');
    }
    
    confirmDeleteTask(taskId = null) {
        const id = taskId || this.currentTaskId;
        this.pendingAction = () => this.deleteTask(id);
        this.confirmMessage.textContent = 'Are you sure you want to delete this task? This action cannot be undone.';
        this.confirmModal.classList.add('show');
    }
    
    async deleteTask(taskId) {
        try {
            const response = await fetch(`/history/${taskId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete task');
            }
            
            this.hideConfirmModal();
            this.hideTaskDetailsModal();
            this.showSuccess('Task deleted successfully');
            this.loadHistory(); // Refresh history
            
        } catch (error) {
            console.error('Error deleting task:', error);
            this.showError('Failed to delete task');
        }
    }
    
    confirmClearHistory() {
        this.pendingAction = () => this.clearHistory();
        this.confirmMessage.textContent = 'Are you sure you want to clear all task history? This action cannot be undone.';
        this.confirmModal.classList.add('show');
    }
    
    async clearHistory() {
        try {
            const response = await fetch('/history/clear', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to clear history');
            }
            
            this.hideConfirmModal();
            this.showSuccess('History cleared successfully');
            this.loadHistory(); // Refresh history
            
        } catch (error) {
            console.error('Error clearing history:', error);
            this.showError('Failed to clear history');
        }
    }
    
    hideConfirmModal() {
        this.confirmModal.classList.remove('show');
        this.pendingAction = null;
    }
    
    executePendingAction() {
        if (this.pendingAction) {
            this.pendingAction();
        }
    }
    
    getTimeAgo(date) {
        const now = new Date();
        const taskDate = new Date(date);
        
        // Calculate the difference in milliseconds
        const diffInMs = now - taskDate;
        const diffInSeconds = Math.floor(diffInMs / 1000);
        
        // Handle negative differences (future dates) or very small differences
        if (diffInSeconds < 0 || diffInSeconds < 5) {
            return 'Just now';
        }
        
        if (diffInSeconds < 60) {
            return `${diffInSeconds} seconds ago`;
        } else if (diffInSeconds < 3600) {
            const minutes = Math.floor(diffInSeconds / 60);
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        } else if (diffInSeconds < 86400) {
            const hours = Math.floor(diffInSeconds / 3600);
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        } else {
            const days = Math.floor(diffInSeconds / 86400);
            return `${days} day${days > 1 ? 's' : ''} ago`;
        }
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + '...';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CrewAIManager();
});

// Add some utility functions for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling behavior
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add keyboard shortcuts info
    const shortcuts = document.querySelector('.shortcuts');
    if (shortcuts) {
        shortcuts.addEventListener('click', () => {
            alert('Keyboard Shortcuts:\n\n• Ctrl + Enter: Start crew\n• Escape: Close modals\n• Tab: Navigate between elements');
        });
    }
    
    // Add auto-save functionality (optional)
    const promptInput = document.getElementById('promptInput');
    if (promptInput) {
        // Save to localStorage on input
        promptInput.addEventListener('input', () => {
            localStorage.setItem('crewai-prompt', promptInput.value);
        });
        
        // Load from localStorage on page load
        const savedPrompt = localStorage.getItem('crewai-prompt');
        if (savedPrompt) {
            promptInput.value = savedPrompt;
            // Trigger character count update
            promptInput.dispatchEvent(new Event('input'));
        }
    }
    
    // Add confirmation before leaving if crew is running
    window.addEventListener('beforeunload', (e) => {
        const manager = window.crewAIManager;
        if (manager && manager.isRunning) {
            e.preventDefault();
            e.returnValue = 'Crew is currently running. Are you sure you want to leave?';
        }
    });
});

// Make the manager globally accessible for debugging
window.CrewAIManager = CrewAIManager;