from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
import os
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class CodingAssistant():
    """CodingAssistant crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    llm_openrouter=LLM(
        model="openrouter/cohere/north-mini-code:free",
        api_key= os.getenv("OPENROUTER_API_KEY")
    )

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'], # type: ignore[index]
            llm=self.llm_openrouter ,
            verbose=True
        )

    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CodingAssistant crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
