import os, argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", help="which UnMicst tool?", default = 'unmicst-solo')
    parser.add_argument("imagePath", help="path to the .tif file")
    parser.add_argument("--model",  help="type of model. For example, nuclei vs cytoplasm")
    parser.add_argument("--outputPath", help="output path of probability map")
    parser.add_argument("--channel", help="channel to perform inference on",  nargs = '+', type=int, default=[1])
    parser.add_argument("--classOrder", help="background, contours, foreground", type = int, nargs = '+', default=-1)
    parser.add_argument("--mean", help="mean intensity of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--std", help="mean standard deviation of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
                        default=1)
    parser.add_argument("--stackOutput", help="save probability maps as separate files", action='store_true')
    parser.add_argument("--GPU", help="explicitly select GPU", type=int, default = 0)
    parser.add_argument("--outlier",
                        help="map percentile intensity to max when rescaling intensity values. Max intensity as default",
                        type=float, default=-1)
    args = parser.parse_args()


    scriptPath =os.path.dirname(os.path.realpath(__file__))


cmd="python " + scriptPath + os.sep

channel = args.channel
classOrder = args.classOrder
GPU = args.GPU


channel[:] = [number - 1 for number in channel]
if classOrder != -1:
    classOrder[:] = [number - 1 for number in classOrder]
GPU = GPU -1	

if args.tool == 'unmicst-duo':
    cmd = cmd +  "UnMicst2.py "
    if len(args.channel)==2:
        channel = str(channel[0]) +" " + str(channel[1])
    else:
        channel = str(channel[0])
elif args.tool == 'unmicst-legacy':
    cmd = cmd + "UnMicst.py "
    channel = str(channel[0])
    print('')
    print("WARNING! YOU HAVE OPTED TO USE UNMICST legacy, WHICH IS GETTING TIRED AND OLD. CONSIDER USING unmicst-solo OR unmicst-duo (IF YOU ALSO HAVE A NUCLEAR ENVELOPE STAIN")
    print('')
elif args.tool == 'UnMicstCyto2':
    cmd = cmd + "UnMicstCyto2.py "
    channel = str(channel[0])
else:
    cmd = cmd + "UnMicst1-5.py "
    channel = str(channel[0])
    print('')
    print(
        "WARNING! USING unmicst-solo AS DEFAULT. THIS MODEL HAS BEEN TRAINED ON MORE TISSUE TYPES. IF YOU WANT THE LEGACY MODEL, USE --tool unmicst-legacy")
    print('')

cmd = cmd + " " + args.imagePath
cmd = cmd + " --channel " + str(channel)
cmd = cmd + " --outputPath " + str(args.outputPath)
cmd = cmd + " --mean " + str(args.mean)
cmd = cmd + " --std " + str(args.std)
cmd = cmd + " --scalingFactor " + str(args.scalingFactor)
cmd = cmd + " --GPU " + str(GPU)
cmd = cmd + " --outlier " + str(args.outlier)

if args.stackOutput:
    cmd = cmd + " --stackOutput "

if args.model:
    cmd = cmd + " --model " + str(args.model)

if args.classOrder == -1:
    cmd = cmd
else:
    cmd = cmd + " --classOrder " + str(classOrder[0]) + " " + str(classOrder[1]) + " " + str(
        classOrder[2])



print(cmd)
os.system(cmd)