"""
FastAPI server for reward model inference.
Serves reward scores for instruction-completion pairs.
"""

import os
import torch
import torch.nn as nn
import uvicorn
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer

# ================= Configuration =================
# Model path - update this to your model location
MODEL_PATH = "./models/ultrarm-13b"

# GPU settings - adjust based on your hardware
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Server configuration
PORT = 9002
MAX_CONCURRENT_REQUESTS = 2

# UltraRM template - strictly follows the required format
ULTRARM_TEMPLATE = """Human: {instruction}

Assistant: {completion}"""
# =================================================


class LlamaRewardModel(PreTrainedModel):
    """
    Custom Llama-based reward model class.
    Matches the UltraRM architecture.
    """
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        # Apply sigmoid and extract last token reward
        rewards = torch.sigmoid(rewards)
        ends = attention_mask.sum(dim=1, keepdim=True) - 1
        ends = ends.to(rewards.device)
        rewards = torch.gather(rewards, 1, ends)
        return rewards


# Global model and tokenizer
model = None
tokenizer = None
gpu_lock = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class RewardRequest(BaseModel):
    """Request schema for reward model API."""
    instruction: str
    completion: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager for model loading and cleanup.
    """
    global model, tokenizer

    # Startup - Load model
    print("Cleaning up distributed environment variables to force single-node mode...")
    dist_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    for v in dist_vars:
        if v in os.environ:
            print(f"   - Removing {v}")
            del os.environ[v]

    print(f"Loading UltraRM model from {MODEL_PATH}...")
    try:
        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.bos_token

        # Load model with device_map="auto" for multi-GPU support
        model = LlamaRewardModel.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model.eval()
        print("Model loaded successfully!")
        print(f"   Device Map: {getattr(model, 'hf_device_map', 'Single GPU')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    yield  # Server running

    # Shutdown - Cleanup
    print("Shutting down server...")
    if model:
        del model
    if tokenizer:
        del tokenizer
    torch.cuda.empty_cache()


# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


@app.post("/get_reward")
async def get_reward_endpoint(request: RewardRequest):
    """
    Get reward score for an instruction-completion pair.

    Args:
        request: Contains instruction and completion

    Returns:
        JSON with "score" field containing the reward value
    """
    async with gpu_lock:
        try:
            # Truncate instruction to prevent excessively long prompts
            instr_truncated = request.instruction[:8192]

            # Apply the UltraRM template
            text = ULTRARM_TEMPLATE.format(
                instruction=instr_truncated,
                completion=request.completion
            )

            # Tokenize with max_length=16092 as specified
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=16092,
                padding=True,
                truncation=True,
            )

            # Move to model's device
            target_device = model.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                reward_tensor = model(**inputs).squeeze(-1)
                reward_val = reward_tensor.detach().cpu().item()

            return {"score": reward_val}

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="CUDA Out of Memory")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
