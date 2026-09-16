from src.data_loader import get_emotion_from_filename, get_actor_from_path

def test_get_emotion_from_filename():
    assert get_emotion_from_filename("03-01-01-01-01-01-01.wav") == "neutral"
    assert get_emotion_from_filename("03-01-03-01-01-01-01.wav") == "happy"
    assert get_emotion_from_filename("03-01-08-01-01-01-01.wav") == "surprised"

def test_get_actor_from_path():
    assert get_actor_from_path("/path/to/data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav") == "Actor_01"
