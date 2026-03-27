# Qwen3 Chat Template Example

This folder shows how to add a new role (`data`) to the Qwen3 chat template.

## Files

- `qwen3_org.jinja`: Original Qwen3 chat template (supports `system`, `user`, `assistant`, `tool` roles)
- `qwen3_new.jinja`: Modified template with added `data` role support

## Key Changes

The main difference is adding support for the `data` role in the message processing loop:

**In `qwen3_new.jinja` (added after line 26):**
```jinja
{%- elif message.role == "data" %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
```

## How to Add a New Role

1. **Identify the message processing loop**: Find the `{%- for message in messages %}` loop in your template.

2. **Add a new condition**: Insert an `elif` clause to handle your new role:
   ```jinja
   {%- elif message.role == "your_role_name" %}
       {{- '<|im_start|>your_role_name\n' + message.content + '<|im_end|>\n' }}
   ```

3. **Follow the same format**: Use the same start/end tokens (`<|im_start|>`, `<|im_end|>`) as other roles in the template.
