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
        args.video_paths, args.file_with_video_paths, to_shuffle=False
    )

    df = pd.read_csv("train_complete_10.csv")

    print(f"The number of specified videos: {len(df)}")

    # labels_df = pd.read_csv("../ML_proyecto_3/datasets/training_id.csv")

    df = df.iloc[:, 1:]
    print(df)
    # all_features_flatten = []
    all_features_mean = []
    labels = []
    ids = []
    j = 0
    # 461 for s3d, 0 for r21d, 251 for vggish
    # video 258, 310, 350, 450 are falling. 2 missing groups. training s3d
    # video  are falling. 0 missing groups. training r21d
    # video 356 are falling. 1 missing groups. val
    limit = 10
    # RuntimeError: Non-empty 4D data tensor expected but got a tensor with sizes [3, 0, 1, 1]
    for i in tqdm(range(0, len(df)), initial=j):
        video_path = video_paths[j]
        print("-------------------------------------------------------------")
        print(f"Extracting for {video_path}")
        features_dict = extractor.extract(video_path)  # note the `_` in the method name
        # youtube_id = video_path[
        #     video_path.rfind("/") + 1 : video_path.rfind("/") + 12
        # ]
        youtube_id = df.iloc[[i], 0].item()
        label = df.iloc[[i], 2].item()
        labels.append(label)
        ids.append(youtube_id)
        for k, v in features_dict.items():
            print(k)
            print(v.shape)
            print(v)
            # Flattening with column major, F
            # all_features_flatten.append(v.flatten("F"))
            all_features_mean.append(v.mean(axis=0))

            if j % limit == 0:
                all_features_mean = np.array(all_features_mean)
                DF = pd.DataFrame(all_features_mean)
                DF["label"] = labels
                DF["youtube_id"] = ids
                filename = (
                    "datasets/training_"
                    + args.feature_type
                    + "_all_"
                    + str(j)
                    + "_mean"
                    + ".csv"
                )
                DF.to_csv(filename)

                all_features_mean = []
                labels = []
                ids = []
                DF = 0
            j += 1

    # yep, it is this simple!


if __name__ == "__main__":
    args_cli = OmegaConf.from_cli()
    main(args_cli)
