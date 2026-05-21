"""
Comet - Main entry point.
Wires browser-use + Gemini 2.5 + Chrome persistent profile
+ Vision + Memory + Filesystem tools into one unified agent.
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.llm.google.chat import ChatGoogle

from comet.utils.logger import CometLogger
from comet.utils.chrome_profile import get_persistent_context_kwargs
from comet.agent.memory import CometMemory
from comet.agent.vision import VisionBrain
from comet.tools.filesystem import FileSystemTools
from comet.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    MEMORY_DIR, LOGS_DIR,
    MAX_REACT_ITERATIONS,
)

console = Console()


class CometAgent:
    """
    Comet = browser-use Agent
            + Chrome persistent profile   (zero 2FA)
            + Gemini 2.5 Pro              (reasoning + vision)
            + ChromaDB memory             (long-term context)
            + FileSystemTools             (Excel / Word / PDF)
    """

    def __init__(self):
        self.logger = CometLogger(log_dir=LOGS_DIR)
        self.memory = CometMemory(
            logger     = self.logger,
            memory_dir = str(MEMORY_DIR),
        )
        self.vision = VisionBrain(
            logger  = self.logger,
            api_key = GEMINI_API_KEY,
            model   = GEMINI_MODEL,
        )
        self.fs = FileSystemTools(logger=self.logger)

        # Use browser-use native Google LLM (compatible with llm.provider check)
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        self.llm = ChatGoogle(model=GEMINI_MODEL)

        ctx_kwargs = get_persistent_context_kwargs()
        profile = BrowserProfile(
            headless        = False,
            executable_path = ctx_kwargs.get("executable_path"),
        )
        self.browser = BrowserSession(browser_profile=profile)

    async def run(self, task: str) -> str:
        console.print(Panel(
            f"[bold magenta]Tache :[/]\n{task}",
            title="[bold]COMET - Demarrage[/]",
            border_style="magenta",
        ))

        memory_ctx    = self.memory.get_context_for_prompt(task)
        enriched_task = (
            f"{task}\n\n--- CONTEXTE MEMOIRE ---\n{memory_ctx}"
            if memory_ctx != "Nouvelle session."
            else task
        )

        agent = Agent(
            task                 = enriched_task,
            llm                  = self.llm,
            browser_session      = self.browser,
            max_actions_per_step = MAX_REACT_ITERATIONS,
        )

        try:
            result  = await agent.run()
            summary = str(result)
            self.memory.save_result(task, summary)
            self.logger.success(f"Tache terminee : {summary[:200]}")
            return summary
        except Exception as e:
            self.logger.error(f"Erreur agent : {e}")
            return f"ERREUR : {e}"
        finally:
            await self.browser.stop()


async def main():
    console.print(Panel(
        "[bold cyan]COMET[/] - Agent de Navigation Web IA\n"
        "[dim]browser-use + Gemini 2.5 Pro + Chrome Profile + Vision + Memory[/]",
        border_style="cyan",
        title="Bienvenue",
    ))
    agent = CometAgent()
    while True:
        console.print("\n[bold]Entrez votre tache[/] (ou [red]exit[/] pour quitter) :")
        task = Prompt.ask("[bold cyan]Comet[/]").strip()
        if task.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Au revoir ![/]")
            break
        if not task:
            continue
        result = await agent.run(task)
        console.print(Panel(f"[green]{result[:500]}[/]", title="Resultat", border_style="green"))


if __name__ == "__main__":
    asyncio.run(main())
