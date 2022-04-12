import argparse

from tqdm import tqdm

from src.data import get_famous_people_zip, get_ibug_zip
from src.matcher import Matcher


def main_famous_people(headless: bool):
    matcher = Matcher(headless=headless)
    famous_people_zip = get_famous_people_zip()
    for face, face_name in (pbar := tqdm(list(famous_people_zip))):
        pbar.set_description_str('running for: %s' % face_name)
        matcher.match(face)
        matcher.clear()


def main_ibug_faces(headless: bool):
    matcher = Matcher(headless=headless)
    ibug_zip = get_ibug_zip()
    for name, face in (pbar := tqdm(list(ibug_zip))):
        pbar.set_description_str('running for: %s' % name)
        matcher.match(face)
        matcher.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # main_ibug_faces(args.headless)
    from src.make_smpls_embeddings import make_smpls_embeddings
    make_smpls_embeddings()
