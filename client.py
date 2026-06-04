import asyncio
import os
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to MCP server
        """

        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")

        if not (is_python or is_js):
            raise ValueError(
                "Server script must be a .py or .js file"
            )

        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()

        print(
            "\nConnected to server with tools:",
            [tool.name for tool in response.tools]
        )

    async def process_query(self, query: str) -> str:
        """
        Process query using Groq + MCP tools
        """

        response = await self.session.list_tools()

        available_tools = []

        for tool in response.tools:
            available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )

        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        final_text = []

        while True:

            completion = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
            )

            assistant_message = completion.choices[0].message

            if assistant_message.content:
                final_text.append(assistant_message.content)

            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                tool_args = json.loads(
                    tool_call.function.arguments
                )

                result = await self.session.call_tool(
                    tool_name,
                    tool_args
                )

                tool_result_text = ""

                if hasattr(result, "content"):
                    for item in result.content:
                        if hasattr(item, "text"):
                            tool_result_text += item.text
                        else:
                            tool_result_text += str(item)
                else:
                    tool_result_text = str(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text,
                    }
                )

        return "\n".join(final_text)

    async def chat_loop(self):
        """
        Interactive chat
        """

        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)

                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python client.py <path_to_server_script>"
        )
        sys.exit(1)

    server_script_path = (
        " ".join(sys.argv[1:])
        .strip()
        .strip('"')
    )

    client = MCPClient()

    try:
        await client.connect_to_server(
            server_script_path
        )

        await client.chat_loop()

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())