"""
Multi-Agent Systems with CrewAI and LangGraph
==============================================

Complete implementation of multi-agent systems for:
- Research teams
- Customer support
- Code generation
- Arabic content creation
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


# ============================================================================
# Part 1: CrewAI-Style Multi-Agent System
# ============================================================================

@dataclass
class AgentRole:
    """Definition of an agent's role"""
    name: str
    goal: str
    backstory: str
    skills: List[str] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = True


@dataclass
class Task:
    """Task to be executed by an agent"""
    description: str
    expected_output: str
    agent: Optional[str] = None
    context: Optional[List[str]] = None
    async_execution: bool = False
    output_json: Optional[Dict] = None
    output_file: Optional[str] = None
    callback: Optional[str] = None


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"


class MultiAgent:
    """
    Multi-agent system implementation
    - Role-based agents
    - Task delegation
    - Sequential and parallel execution
    - Context sharing
    """
    
    def __init__(self):
        self.agents = {}
        self.tasks = []
        self.task_results = {}
        self.process_type = "sequential"  # sequential or parallel
    
    def add_agent(self, role: AgentRole):
        """Add an agent to the crew"""
        self.agents[role.name] = {
            'role': role,
            'state': AgentState.IDLE,
            'current_task': None,
            'completed_tasks': []
        }
        print(f"✓ Agent added: {role.name}")
    
    def add_task(self, task: Task):
        """Add a task to be executed"""
        self.tasks.append(task)
        print(f"✓ Task added: {task.description[:50]}...")
    
    def execute_task(self, task: Task) -> Dict:
        """
        Execute a single task
        In production, this would call LLM
        """
        agent_name = task.agent
        
        if agent_name not in self.agents:
            return {'error': f'Agent {agent_name} not found'}
        
        # Update agent state
        self.agents[agent_name]['state'] = AgentState.EXECUTING
        self.agents[agent_name]['current_task'] = task.description
        
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"Task: {task.description}")
        print(f"{'='*60}")
        
        # Simulate task execution (in production: call LLM)
        result = self._simulate_llm_execution(task, agent_name)
        
        # Update agent state
        self.agents[agent_name]['state'] = AgentState.COMPLETED
        self.agents[agent_name]['completed_tasks'].append(task.description)
        self.agents[agent_name]['current_task'] = None
        
        # Store result
        self.task_results[task.description] = result
        
        return result
    
    def _simulate_llm_execution(self, task: Task, agent_name: str) -> str:
        """
        Simulate LLM execution
        In production, replace with actual LLM call
        """
        agent = self.agents[agent_name]['role']
        
        # Build prompt
        prompt = f"""
You are {agent_name}.
{agent.backstory}

Your goal: {agent.goal}

Task: {task.description}

Expected output: {task.expected_output}
"""
        
        # In production: call LLM
        # response = llm.generate(prompt)
        
        # Mock response
        response = f"[{agent_name} completed task: {task.description[:30]}...]"
        
        return {
            'agent': agent_name,
            'task': task.description,
            'result': response,
            'prompt': prompt
        }
    
    def execute_sequential(self) -> List[Dict]:
        """Execute all tasks sequentially"""
        print("\n🚀 Starting Sequential Execution")
        print("=" * 60)
        
        results = []
        
        for task in self.tasks:
            result = self.execute_task(task)
            results.append(result)
            
            # Pass context to next task
            if result.get('result'):
                for next_task in self.tasks[self.tasks.index(task) + 1:]:
                    if next_task.context is None:
                        next_task.context = []
                    next_task.context.append(result['result'])
        
        print("\n" + "=" * 60)
        print("✓ Sequential execution completed")
        print("=" * 60)
        
        return results
    
    def execute_parallel(self, max_concurrent: int = 3) -> List[Dict]:
        """Execute independent tasks in parallel"""
        print("\n🚀 Starting Parallel Execution")
        print("=" * 60)
        
        # Group tasks by agent
        tasks_by_agent = {}
        for task in self.tasks:
            if task.agent not in tasks_by_agent:
                tasks_by_agent[task.agent] = []
            tasks_by_agent[task.agent].append(task)
        
        results = []
        
        # Execute tasks for each agent
        for agent_name, agent_tasks in tasks_by_agent.items():
            print(f"\nExecuting tasks for agent: {agent_name}")
            for task in agent_tasks:
                result = self.execute_task(task)
                results.append(result)
        
        print("\n" + "=" * 60)
        print("✓ Parallel execution completed")
        print("=" * 60)
        
        return results
    
    def get_agent_status(self) -> Dict:
        """Get status of all agents"""
        status = {}
        for name, agent_data in self.agents.items():
            status[name] = {
                'state': agent_data['state'].value,
                'current_task': agent_data['current_task'],
                'completed_tasks': len(agent_data['completed_tasks'])
            }
        return status


# ============================================================================
# Part 2: Pre-built Agent Teams
# ============================================================================

class ResearchTeam(MultiAgent):
    """Research team with specialized agents"""
    
    def __init__(self):
        super().__init__()
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup research team agents"""
        
        # Researcher agent
        researcher = AgentRole(
            name="Senior Research Analyst",
            goal="Find relevant, accurate information on any topic",
            backstory="""You are an expert research analyst with 10 years of experience.
You excel at finding high-quality information from reliable sources.
You always verify facts and cite sources.""",
            skills=["research", "fact-checking", "source evaluation"]
        )
        self.add_agent(researcher)
        
        # Writer agent
        writer = AgentRole(
            name="Content Writer",
            goal="Write compelling, accurate content based on research",
            backstory="""You are an expert content writer with technical knowledge.
You transform complex research into clear, engaging content.
You maintain accuracy while making content accessible.""",
            skills=["writing", "editing", "simplification"]
        )
        self.add_agent(writer)
        
        # Editor agent
        editor = AgentRole(
            name="Senior Editor",
            goal="Ensure content quality, accuracy, and clarity",
            backstory="""You are a meticulous editor with 15 years of experience.
You catch errors, improve clarity, and ensure consistency.
You provide constructive feedback.""",
            skills=["editing", "proofreading", "quality assurance"]
        )
        self.add_agent(editor)
    
    def create_research_workflow(self, topic: str) -> List[Task]:
        """Create research workflow tasks"""
        
        tasks = [
            Task(
                description=f"Research the topic: {topic}. Find key concepts, recent developments, and major players.",
                expected_output="Detailed research findings with sources",
                agent="Senior Research Analyst"
            ),
            Task(
                description="Based on research findings, write a comprehensive article (500-700 words).",
                expected_output="Published-ready article",
                agent="Content Writer",
                context=[]
            ),
            Task(
                description="Review and edit the article for accuracy, clarity, and grammar.",
                expected_output="Final edited article with notes",
                agent="Senior Editor",
                context=[]
            )
        ]
        
        for task in tasks:
            self.add_task(task)
        
        return tasks


class ArabicContentTeam(MultiAgent):
    """Arabic content creation team"""
    
    def __init__(self):
        super().__init__()
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup Arabic content team"""
        
        # Arabic writer
        arabic_writer = AgentRole(
            name="كاتب عربي",  # Arabic Writer
            goal="إنشاء محتوى عربي عالي الجودة",  # Create high-quality Arabic content
            backstory="""أنت كاتب محترف متخصص في اللغة العربية الفصحى.
لديك خبرة 10 سنوات في كتابة المحتوى العربي.
تحرص على الدقة والوضوح والأسلوب الشيق.""",
            skills=["الكتابة العربية", "التحرير", "التدقيق اللغوي"]
        )
        self.add_agent(arabic_writer)
        
        # Translator
        translator = AgentRole(
            name="مترجم",  # Translator
            goal="ترجمة دقيقة من الإنجليزية إلى العربية",  # Accurate translation
            backstory="""أنت مترجم محترف متخصص في الترجمة من الإنجليزية إلى العربية.
تحافظ على المعنى والدقة مع ضمان سلاسة النص العربي.""",
            skills=["الترجمة", "التوطين", "التكيف الثقافي"]
        )
        self.add_agent(translator)
        
        # Reviewer
        reviewer = AgentRole(
            name="مراجع لغوي",  # Language Reviewer
            goal="مراجعة المحتوى العربي للتأكد من جودته",  # Review Arabic content quality
            backstory="""أنت خبير في اللغة العربية وقواعدها.
تراجع المحتوى للتأكد من خلوه من الأخطاء اللغوية والإملائية.""",
            skills=["المراجعة اللغوية", "التدقيق الإملائي", "ضبط الجودة"]
        )
        self.add_agent(reviewer)
    
    def create_arabic_workflow(self, topic: str) -> List[Task]:
        """Create Arabic content workflow"""
        
        tasks = [
            Task(
                description=f"اكتب مقالاً عن: {topic}",
                expected_output="مقال عربي متكامل",
                agent="كاتب عربي"
            ),
            Task(
                description="راجع المقال من الناحية اللغوية والإملائية",
                expected_output="مقال مراجع ومصحح",
                agent="مراجع لغوي",
                context=[]
            )
        ]
        
        for task in tasks:
            self.add_task(task)
        
        return tasks


class CodeGenerationTeam(MultiAgent):
    """Code generation team"""
    
    def __init__(self):
        super().__init__()
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup code generation team"""
        
        # Architect
        architect = AgentRole(
            name="Software Architect",
            goal="Design clean, scalable software architectures",
            backstory="""You are a senior software architect with 15 years of experience.
You design systems that are scalable, maintainable, and robust.
You follow best practices and design patterns.""",
            skills=["architecture", "design patterns", "system design"]
        )
        self.add_agent(architect)
        
        # Developer
        developer = AgentRole(
            name="Senior Developer",
            goal="Write clean, efficient, well-tested code",
            backstory="""You are a senior developer with expertise in multiple languages.
You write clean, efficient code with proper error handling.
You follow coding standards and best practices.""",
            skills=["coding", "testing", "debugging"]
        )
        self.add_agent(developer)
        
        # Reviewer
        reviewer = AgentRole(
            name="Code Reviewer",
            goal="Ensure code quality and best practices",
            backstory="""You are an experienced code reviewer.
You catch bugs, suggest improvements, and ensure code quality.
You provide constructive feedback.""",
            skills=["code review", "quality assurance", "security"]
        )
        self.add_agent(reviewer)
    
    def create_coding_workflow(self, requirement: str) -> List[Task]:
        """Create code generation workflow"""
        
        tasks = [
            Task(
                description=f"Design architecture for: {requirement}",
                expected_output="Architecture design with components and interfaces",
                agent="Software Architect"
            ),
            Task(
                description="Implement the designed architecture in Python",
                expected_output="Complete, working code with tests",
                agent="Senior Developer",
                context=[]
            ),
            Task(
                description="Review the code for quality, security, and best practices",
                expected_output="Code review with suggestions and improvements",
                agent="Code Reviewer",
                context=[]
            )
        ]
        
        for task in tasks:
            self.add_task(task)
        
        return tasks


# ============================================================================
# Part 3: Example Usage and Demos
# ============================================================================

def demo_research_team():
    """Demo: Research team workflow"""
    print("\n" + "=" * 60)
    print("DEMO: Research Team")
    print("=" * 60)
    
    team = ResearchTeam()
    tasks = team.create_research_workflow("The impact of AI on software engineering")
    
    print(f"\nCreated {len(tasks)} tasks for research workflow")
    print(f"Agents: {list(team.agents.keys())}")
    
    # Execute
    results = team.execute_sequential()
    
    # Show results
    print("\n📊 Results Summary:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['agent']}")
        print(f"   Task: {result['task'][:50]}...")
        print(f"   Result: {result['result'][:100]}...")
    
    return results


def demo_arabic_team():
    """Demo: Arabic content team"""
    print("\n" + "=" * 60)
    print("DEMO: Arabic Content Team")
    print("=" * 60)
    
    team = ArabicContentTeam()
    tasks = team.create_arabic_workflow("الذكاء الاصطناعي")  # Artificial Intelligence
    
    print(f"\nCreated {len(tasks)} tasks for Arabic workflow")
    print(f"Agents: {list(team.agents.keys())}")
    
    # Execute
    results = team.execute_sequential()
    
    # Show results
    print("\n📊 Results Summary:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['agent']}")
        print(f"   Task: {result['task'][:50]}...")
        print(f"   Result: {result['result'][:100]}...")
    
    return results


def demo_code_team():
    """Demo: Code generation team"""
    print("\n" + "=" * 60)
    print("DEMO: Code Generation Team")
    print("=" * 60)
    
    team = CodeGenerationTeam()
    tasks = team.create_coding_workflow("REST API for user authentication")
    
    print(f"\nCreated {len(tasks)} tasks for coding workflow")
    print(f"Agents: {list(team.agents.keys())}")
    
    # Execute
    results = team.execute_sequential()
    
    # Show results
    print("\n📊 Results Summary:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['agent']}")
        print(f"   Task: {result['task'][:50]}...")
        print(f"   Result: {result['result'][:100]}...")
    
    return results


def compare_execution_modes():
    """Compare sequential vs parallel execution"""
    print("\n" + "=" * 60)
    print("COMPARISON: Sequential vs Parallel Execution")
    print("=" * 60)
    
    team = MultiAgent()
    
    # Add agents
    for i in range(3):
        agent = AgentRole(
            name=f"Agent_{i}",
            goal=f"Goal {i}",
            backstory=f"Backstory {i}",
            skills=[f"skill_{i}"]
        )
        team.add_agent(agent)
    
    # Add tasks
    for i in range(6):
        task = Task(
            description=f"Task {i}",
            expected_output=f"Output {i}",
            agent=f"Agent_{i % 3}"
        )
        team.add_task(task)
    
    import time
    
    # Sequential
    print("\n⏱️ Sequential Execution:")
    start = time.time()
    team.execute_sequential()
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")
    
    # Reset
    team.tasks = []
    team.task_results = {}
    
    # Parallel
    print("\n⏱️ Parallel Execution:")
    start = time.time()
    team.execute_parallel()
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")
    
    print(f"\nSpeedup: {sequential_time / parallel_time:.2f}x")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Systems Demo")
    print("=" * 60)
    
    # Run demos
    demo_research_team()
    demo_arabic_team()
    demo_code_team()
    compare_execution_modes()
    
    print("\n" + "=" * 60)
    print("✓ All demos completed!")
    print("=" * 60)
