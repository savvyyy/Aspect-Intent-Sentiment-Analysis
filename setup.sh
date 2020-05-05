#install the requirements

pip install -r requirements.txt

pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python $HEAD_DIR/nltk/nltk_script.py