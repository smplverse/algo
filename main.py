import argparse
import time

from tqdm import tqdm
from src.matcher import Matcher
from src.data import get_validation_zip, get_smpls
from src.vgg_face2 import VGGFace2


def main(headless: bool):
    model = VGGFace2()
    for face, face_name in get_validation_zip():
        tic = time.time()
        print('\nrunning for:', face_name)
        matcher_session = Matcher(model, headless, session=True)
        paths, smpls = get_smpls("data/smpls")
        smpls_zip = zip(paths, smpls)
        for _ in tqdm(range(len(smpls))):
            _, smpl = smpls_zip.__next__()
            matcher_session.match(face, smpl)
        toc = time.time()
        print("total time elapsed: %.2fs" % float(toc - tic))
        matcher_session.write_results(face, smpls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(args.headless)
