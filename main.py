import argparse

from tqdm import tqdm

from src.data import get_famous_people_zip, get_ibug_zip
from src.matcher import Matcher


def main_famous_people(headless: bool, model: str):
    matcher = Matcher(headless=headless, model=model)
    famous_people_zip = get_famous_people_zip()
    for face, face_name in (pbar := tqdm(list(famous_people_zip))):
        pbar.set_description_str('running for: %s' % face_name)
        matcher.match(face)
        matcher.clear()


def main_ibug_faces(headless: bool, model: str):
    matcher = Matcher(headless=headless, model=model)
    ibug_zip = get_ibug_zip()
    for name, face in (pbar := tqdm(list(ibug_zip))):
        pbar.set_description_str('running for: %s' % name)
        matcher.match(face)
        matcher.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", type=str, default="resnet100")
    parser.add_argument("--dataset", type=str, default="ibug_faces")
    args = parser.parse_args()

    if args.dataset == "famous_people":
        main_famous_people(args.headless, args.model)
    elif args.dataset == "ibug_faces":
        main_ibug_faces(args.headless, args.model)
    else:
        raise ValueError("Unknown dataset: %s" % args.dataset)
