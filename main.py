from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check


def main(args_cli):
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    print(args)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    if args.on_extraction in ["save_numpy", "save_pickle"]:
        print(f"Saving features to {args.output_path}")
    print("Device:", args.device)

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == "i3d":
        from models.i3d.extract_i3d import ExtractI3D as Extractor
    elif args.feature_type == "r21d":
        from models.r21d.extract_r21d import ExtractR21D as Extractor
    elif args.feature_type == "s3d":
        from models.s3d.extract_s3d import ExtractS3D as Extractor
    elif args.feature_type == "vggish":
        from models.vggish.extract_vggish import ExtractVGGish as Extractor
    # elif args.feature_type == "resnet":
    #     from models.resnet.extract_resnet import ExtractResNet as Extractor
    # elif args.feature_type == "raft":
    #     from models.raft.extract_raft import ExtractRAFT as Extractor
    # elif args.feature_type == "clip":
    #     from models.clip.extract_clip import ExtractCLIP as Extractor
    # elif args.feature_type == "timm":
    #     from models.timm.extract_timm import ExtractTIMM as Extractor
    else:
        raise NotImplementedError(f"Extractor {args.feature_type} is not implemented.")

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(
        args.video_paths, args.file_with_video_paths, to_shuffle=True
    )

    print(f"The number of specified videos: {len(video_paths)}")

    all_features_flatten = []
    all_features_mean = []
    for video_path in tqdm(video_paths):
        print("-------------------------------------------------------------")
        print(f"Extracting for {video_path}")
        features_dict = extractor.extract(video_path)  # note the `_` in the method name
        for k, v in features_dict.items():
            print(k)
            print(v.shape)
            print(v)
            # Flattening with column major, F
            # all_features_flatten.append(v.flatten("F"))
            all_features_mean.append(v.mean(axis=0))
    # all_features_flatten = np.array(all_features_flatten)
    # print(all_features_flatten)
    # DF = pd.DataFrame(all_features_flatten)
    # print(DF)
    # filename = (
    #     "datasets/train_"
    #     + args.feature_type
    #     + "_"
    #     + str(len(video_paths))
    #     + "_flatten"
    #     + ".csv"
    # )
    # DF.to_csv(filename)
    all_features_mean = np.array(all_features_mean)
    DF = pd.DataFrame(all_features_mean)
    # print(all_features_mean)
    # print(DF)
    filename = (
        "datasets/train_"
        + args.feature_type
        + "_"
        + str(len(video_paths))
        + "_mean"
        + ".csv"
    )
    DF.to_csv(filename)

    # yep, it is this simple!


if __name__ == "__main__":
    args_cli = OmegaConf.from_cli()
    main(args_cli)
