import os
import torch
from open_voice.openvoice import se_extractor
from open_voice.openvoice.api import ToneColorConverter
from open_voice.MeloTTS.melo.api import TTS


ckpt_converter = 'open_voice/checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'open_voice/outputs_v2'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
os.makedirs(output_dir, exist_ok=True)


langs={

      'EN-Newest':('EN_NEWEST',0),
      'EN-US':('EN',0),
      'EN-BR':('EN',1),
      'EN_INDIA':('EN',2),
      'EN-AU':('EN',3),
      'EN-Default':('EN',4),
      'ES':('ES',1),
      'FR':('FR',0),
      'ZH':('ZH',1),
      'JP':('JP',0),
      'KR':('KR',0)
}

langs_generic={

      'EN-US':('EN',0),
      'EN-BR':('EN',1),
      'EN_INDIA':('EN',2),
      'EN-AU':('EN',3),
      'EN-Default':('EN',4),
      'ES':('ES',1),
      'FR':('FR',0),
      'ZH':('ZH',1),
      'JP':('JP',0),
      'KR':('KR',0)
}

def find_tune(reference_speaker):
    '''
    fine-tuning model on target voice
    '''
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
    return target_se

def synthesize_speech(language,speaker_key,speaker_id,text,target_se,src_path,save_path):
    '''
    Voice clonning
    '''
    speed = 1.0
    model = TTS(language=language, device=device)
    speaker_key = speaker_key.lower().replace('_', '-')
    source_se = torch.load(f'open_voice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
    model.tts_to_file(text, speaker_id, src_path, speed=speed)
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
          audio_src_path=src_path,
          src_se=source_se,
          tgt_se=target_se,
          output_path=save_path,
          message=encode_message)

def melo_tts(text,language,id,output_path):
      '''
      General TTS
      '''
      speed=1.0
      model = TTS(language=language, device=device)
      model.tts_to_file(text, id, output_path, speed=speed)



def main():
    # Voice Cloning
    output_dir = 'open_voice/output_files'
    src_path = f'{output_dir}/tmp.wav'
    speaker_key = 'EN-US'
    if speaker_key not in langs.keys():
        speaker_key = 'EN-Default'
    language, speaker_id = langs[speaker_key]
    save_path = f'{output_dir}/output_v2_{language}.wav'
    print(save_path)
    text = "Did you ever hear a folk tale about a giant turtle?"
    reference_speaker = 'open_voice/resources/example_reference.mp3'  # This is the voice you want to clone
    target_se = find_tune(reference_speaker)
    synthesize_speech(language, speaker_key, speaker_id, text, target_se, src_path, save_path)

    #General TTS
    speaker_key = 'EN-BR'
    if speaker_key not in langs.keys():
        speaker_key = 'EN-Default'
    language, speaker_id = langs[speaker_key]
    text = "Did you ever hear a folk tale about a giant turtle?"
    output_dir = 'open_voice/output_files'
    output_path = f'{output_dir}/generaic_{language}.wav'
    melo_tts(text, language, speaker_id, output_path)

if __name__ == "__main__":
    main()


