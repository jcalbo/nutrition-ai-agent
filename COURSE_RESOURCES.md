# AI Agents Crash Course: Build with Python & OpenAI

## Environment Setup

1) Sign up to OpenAI and [generate an OpenAI API Key](https://platform.openai.com/api-keys)
2) Copy the contents of [.env.template](.env.template) to a file called `.env` and add your api key to to it. You can use the UI or execute these commands in the terminal:
```
cp .env.template .env
code .env
# Add in your OpenAI API Key
```
3) Open our first notebook, [notebooks/simplest_agent.ipynb) and execute the first cell

## Our First Agent

* [Understanding async/await in Python](https://realpython.com/async-io-python/)
* [OpenAI Tracing UI](https://platform.openai.com/logs?api=traces)

## MCP
* [Smithery.ai MCP Hub](https://smithery.ai/)
* [Exa Search](https://exa.ai/)
* [OpenAI Tools Documentation](https://openai.github.io/openai-agents-python/tools/) for the exercise

## Running the Chatbot
One-time setup
```
cd chatbot && chainlit run 4_authentication.py --port 10000 --host 0.0.0.0
```

The URL for our demo logo: 
```
https://upload.wikimedia.org/wikipedia/commons/e/e3/Udemy_logo.svg
```

### Adding Authentication
1) Generate a Chainlit Secret: `chainlit create-secret`
2) Add the secret to `.env`
3) Pick an arbitrary username and password and fill in the `CHAINLIT_USERNAME` and `CHAINLIT_PASSWORD` values in `.env`

## Deployment
Here is the command:
```
chainlit run chatbot/4_authentication.py --port 10000 --host 0.0.0.0

# Datasets
Here are the originals of the datasets we use in the course:
* [Calorie Database](https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams)
* [Nutritionist Q&A](https://huggingface.co/datasets/RaniRahbani/nutritionist)
