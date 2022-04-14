import argparse

from tqdm import tqdm

from src.data import get_famous_people_zip, get_ibug_zip
from src.matcher import Matcher
from src.make_smpls_embeddings import make_smpls_embeddings
from src.generate_image_hash import generate_image_hash


def main_famous_people(headless: bool, model: str):
    matcher = Matcher(headless=headless, model=model)
    famous_people_zip = get_famous_people_zip()
    for name, face in (pbar := tqdm(list(famous_people_zip))):
        pbar.set_description_str('running for: %s' % name)
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
    dsc = "Runs the matcher on chosen dataset against smplverse pieces"
    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument(
        "--generate-hash",
        action="store_true",
        help="Generate hash for all faces in dataset",
    )
    parser.add_argument('--collection-size', type=int, required=True)
    parser.add_argument('--a', type=int, default=256)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="include flag to skip displaying the images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet100",
        help="the model to use, either resnet100 or vggface2",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ibug_faces",
        help="ibug_faces or famous_people",
    )
    parser.add_argument(
        "--make-embeddings",
        action="store_true",
        help="include to create embeddings with the given model",
    )
    args = parser.parse_args()

    if args.generate_hash:
        generate_image_hash(args.collection_size, args.a)
        exit()

    if args.make_embeddings:
        make_smpls_embeddings(args.model)

    if args.dataset == "famous_people":
        main_famous_people(args.headless, args.model)
    elif args.dataset == "ibug_faces":
        main_ibug_faces(args.headless, args.model)
    else:
        raise ValueError("Unknown dataset: %s" % args.dataset)
