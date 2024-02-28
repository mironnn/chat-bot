#!/bin/sh

apt update
apt install -y curl

./bin/ollama serve &

sleep 5

curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2:70b-chat-q4_1"}'

sleep 30

tail -f /dev/null
