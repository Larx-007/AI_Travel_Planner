import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.agentic_workflow import GraphBuilder


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(query)
        graph = GraphBuilder(model_provider='groq')
        react_app = graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open('my_graph.png', 'wb') as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        messages = {'messages': [query.question]}
        output = react_app.invoke(messages)

        if isinstance(output, dict) and 'messages' in output:
            final_output = output['messages'][-1].content  # Last AI response
        else:
            final_output = str(output)

        return {'answer': final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
