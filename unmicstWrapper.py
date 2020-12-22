import os, argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", help="which UnMicst tool?", default = 'UnMicst')
    parser.add_argument("imagePath", help="path to the .tif file")
    parser.add_argument("--model",  help="type of model. For example, nuclei vs cytoplasm")
    parser.add_argument("--outputPath", help="output path of probability map")
    parser.add_argument("--channel", help="channel to perform inference on",  nargs = '+', default=[0])
    parser.add_argument("--classOrder", help="background, contours, foreground", type = int, nargs = '+', default=-1)
    parser.add_argument("--mean", help="mean intensity of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--std", help="mean standard deviation of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
                        default=1)
    parser.add_argument("--stackOutput", help="save probability maps as separate files", action='store_true')
    parser.add_argument("--GPU", help="explicitly select GPU", type=int, default = -1)
    parser.add_argument("--outlier",
                        help="map percentile intensity to max when rescaling intensity values. Max intensity as default",
                        type=float, default=-1)
    args = parser.parse_args()


    scriptPath =os.path.dirname(os.path.realpath(__file__))

cmd="python " + scriptPath + os.sep

channel = args.channel

if args.tool == 'UnMicst2':
    cmd = cmd +  "UnMicst2.py "
    if len(args.channel)==2:
        channel = str(channel[0]) +" " + str(channel[1])
    else:
        channel = str(channel[0])
elif args.tool == 'UnMicst1-5':
    cmd = cmd + "UnMicst1-5.py "
    channel = str(channel[0])
elif args.tool == 'UnMicstCyto2':
    cmd = cmd + "UnMicstCyto2.py "
    channel = str(channel[0])
else:
    cmd = cmd + "UnMicst.py "
    channel = str(channel[0])


cmd = cmd + " " + args.imagePath
cmd = cmd + " --channel " + str(channel)
cmd = cmd + " --outputPath " + str(args.outputPath)
cmd = cmd + " --mean " + str(args.mean)
cmd = cmd + " --std " + str(args.std)
cmd = cmd + " --scalingFactor " + str(args.scalingFactor)
cmd = cmd + " --GPU " + str(args.GPU)
cmd = cmd + " --outlier " + str(args.outlier)

if args.stackOutput:
    cmd = cmd + " --stackOutput "

if args.model:
    cmd = cmd + " --model " + str(args.model)

if args.classOrder == -1:
    cmd = cmd
else:
    cmd = cmd + " --classOrder " + str(args.classOrder[0]) + " " + str(args.classOrder[1]) + " " + str(
        args.classOrder[2])



print(cmd)
os.system(cmd)