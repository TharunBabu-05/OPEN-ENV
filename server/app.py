from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import ESGEnvironment
from tasks import TASKS


app = FastAPI(title="ESG Compliance Environment API")

# Default task so /step works after startup without explicit task selection.
env = ESGEnvironment(task_config=TASKS["basic_compliance"])


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=8)


@app.get("/")
def root():
    return {
        "message": "ESG Environment Running",
        "available_tasks": list(TASKS.keys()),
        "default_task": "basic_compliance",
    }


@app.post("/reset")
def reset(task_id: str = "basic_compliance", seed: int = 42):
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}",
        )

    global env
    env = ESGEnvironment(task_config=TASKS[task_id], seed=seed)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(payload: StepRequest):
    try:
        observation, reward, terminated, truncated, info = env.step(payload.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


@app.get("/state")
def state():
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
