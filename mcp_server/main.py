# This is the main entry point for the MCP server
# from mcp_server.server import mcp
# from mcp_server.tools import *

from fastmcp import FastMCP, Context

mcp = FastMCP(name="PyCliffordMCP")

@mcp.tool()
async def summarize_text(text: str, ctx: Context) -> str:
    """Summarize the provided text using the client's LLM."""
    response = await ctx.sample(
        messages=f"Summarize the following text:\n{text}",
        system_prompt="You are a helpful assistant that summarizes text.",
        temperature=0.5,
        max_tokens=150,
    )
    return response.text


# Run the MCP server
if __name__ == "__main__":
    mcp.run() 