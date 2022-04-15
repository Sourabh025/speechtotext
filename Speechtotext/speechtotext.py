import torch
import zipfile
import torchaudio
from glob import glob

import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.post("/root")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}', "ab") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"file_name": file.filename}


device = torch.device('cpu')


model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en',  # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file, any format compatible with TorchAudio (soundfile backend)
# torch.hub.download_url_to_file('https://www.signalogic.com/melp/EngSamples/eng_m2.wav',
#                                dst='speech_orig.wav', progress=True)
test_files = glob('eng_f2.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
# print(output)


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1


@app.get("/result")
def result():
    l = []

    for example in output:
        l.append(decoder(example.cpu()))

    result = listToString(l)
    return {"Data": result}
