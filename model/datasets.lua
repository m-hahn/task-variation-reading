--"/u/scr/mhahn/"
assert(SCRATCH_PATH ~= nil)
if DATASET == 1 then -- DEPRECATED
  crash()
--  DATASET_DIR = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/"
--  readDict.corpusDir = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questionsChars/training/"
--  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questionsChars/num2Chars"
elseif DATASET == 2 then -- DEPRECATED
  crash()
--  DATASET_DIR = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/"
--  readDict.corpusDir = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/training4/"
--  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/num2Chars"
--  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/listOfFilesToRead-training4.txt"
elseif DATASET == 3 then -- the right one for deepmind for autoencoding (however anonymized)
  DATASET_DIR = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/"
  readDict.corpusDir = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/training2/"
  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/num2Chars"
  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/listOfFilesToRead.txt"
  numericalAnnotationDir = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/annotation/surp-pg-test-langmod-20-200-0.7-100e0/" -- TODO surprisal
elseif DATASET == 4 then -- PTB
  DATASET_DIR = "/disk/scratch2/s1582047/ptb/" -- this is where annotation files will the created
  readDict.corpusDir = "/disk/scratch2/s1582047/ptb/ptb2.0Num/"
  readDict.dictLocation = "/disk/scratch2/s1582047/ptb/num2Chars"
  LIST_OF_FILES_TO_READ = "/disk/scratch2/s1582047/ptb/fileListTesting.txt"  
elseif DATASET == 5 then -- qa
  if params.TASK ~= 'neat-qa' then
      crash()
  end
  DATASET_DIR = SCRATCH_PATH.."deepmind-qa/cnn/training/noentities/"
  readDict.corpusDir = SCRATCH_PATH.."deepmind-qa/cnn/training/noentities/"
  readDict.dictLocation = "/u/scr/mhahn/num2CharsNoEntities-cnn"
  if false then
    LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/cnn-index.txt"
  else
    LIST_OF_FILES_TO_READ =  SCRATCH_PATH.."deepmind-qa/cnn/training/noentities/correctlyAnswered-500.txt"
--     LIST_OF_FILES_TO_READ =  SCRATCH_PATH.."deepmind-qa/cnn/training/noentities/correctlyAnswered-500-scores-cleaned.txt"
  end
  print("Please consider. Reading files from "..LIST_OF_FILES_TO_READ)
elseif DATASET == 6 then -- qa
  if params.TASK ~= 'neat-qa' then
      crash()
  end
  DATASET_DIR = "/u/scr/mhahn/testdata/inference/"
  readDict.corpusDir = "/u/scr/mhahn/testdata/inference/"
  readDict.dictLocation = "/u/scr/mhahn/testdata/inference/num2Chars"
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/testdata/inference/listOfFiles.txt"
elseif DATASET == 7 then -- DEPRECATED
  crash()
--  DATASET_DIR = "/disk/scratch2/s1582047/dundeetreebank/parts/PART1/" -- this is where annotation files will the created
--  corpusDir = "/disk/scratch2/s1582047/dundeetreebank/parts/PART1Num/"
--  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/num2Chars"
--  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/dundeetreebank/PART1List.txt"
elseif DATASET == 8 then -- DEEPMIND CORPUS DEANONYMIZED
  DATASET_DIR = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/withEntities/"
  readDict.corpusDir = "/disk/scratch2/s1582047/deepmind/rc-data/dailymail/questions/withEntities/numerified/"
  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/num2CharsWithEntities"
  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/listOfFilesToRead.txt"
elseif DATASET == 9 then -- DUNDEE DEANONYMIZED
  DATASET_DIR = "/disk/scratch2/s1582047//dundeetreebank/parts/PART1/"
  readDict.corpusDir = "/disk/scratch2/s1582047/dundeetreebank/parts/PART1NumEntities/"
  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/num2CharsWithEntities"
  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/dundeetreebank/PART1List.txt"
  numericalAnnotationDir = "/disk/scratch2/s1582047/dundeetreebank/parts/PART1PredefinedNumericalWithSurp/"
elseif DATASET == 10 then -- DUNDEE TEST
  DATASET_DIR = "/disk/scratch2/s1582047//dundeetreebank/parts/PART2/"
  readDict.corpusDir = "/disk/scratch2/s1582047/dundeetreebank/parts/PART2NumEntities/"
  readDict.dictLocation = "/disk/scratch2/s1582047/deepmind/num2CharsWithEntities"
  LIST_OF_FILES_TO_READ =  "/disk/scratch2/s1582047/dundeetreebank/PART2List.txt"
  numericalAnnotationDir = "/disk/scratch2/s1582047/dundeetreebank/parts/PART2PredefinedNumericalWithSurp/"
elseif DATASET == 11 then -- qa
  if params.TASK ~= 'combined' and params.TASK ~= 'langmod' then
--      print(params.TASK)
      crash()
  end
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/cnn/training/noentities/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/cnn/training/noentities/texts/"
  readDict.dictLocation = "/u/scr/mhahn/num2CharsNoEntities-cnn" --CNN
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/cnn-index.txt"
elseif DATASET == 12 then -- our eyetracking dataset
  if params.TASK ~= 'neat-qa' then
      crash()
  end
  DATASET_DIR = "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/questions/"
  readDict.corpusDir = "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/questions/numerified-anonymous/"
  readDict.dictLocation = "/u/scr/mhahn/num2CharsNoEntities-cnn" --CNN
  LIST_OF_FILES_TO_READ =  "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/fileList.txt"
elseif DATASET == 13 then -- qa -- the joint dataset
  if params.TASK ~= 'neat-qa' and params.TASK ~= 'combined' and params.TASK ~= 'langmod' then
      crash()
  end
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/joint/training/noentities/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/joint/training/noentities/"
  readDict.dictLocation = "/afs/cs.stanford.edu/u/mhahn/num2CharsNoEntities-deepmind-joint" --JOINT
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/deepmind-joint-index.txt"
elseif DATASET == 14 then -- qa the joint dataset (for language modeling/NEAT)
  if params.TASK ~= 'combined' and params.TASK ~= 'langmod' then
--      print(params.TASK)
      crash()
  end
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/joint/training/noentities/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/joint/training/noentities/texts/"
  readDict.dictLocation = "/afs/cs.stanford.edu/u/mhahn/num2CharsNoEntities-deepmind-joint" --JOINT
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/deepmind-joint-index.txt"
elseif DATASET == 15 then -- qa -- the SAMPLE from the joint dataset (100 texts from CNN+Dailymail)
  if params.TASK ~= 'neat-qa' then
      crash()
  end
  assert(DOING_EVALUATION_OUTPUT)
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/joint/training/noentities-sample/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/joint-sample/training/noentities-cnn/"
  readDict.dictLocation = "/u/scr/mhahn/num2CharsNoEntities-cnn" --CNN
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/deepmind-sample-dev.txt"
elseif DATASET == 16 then -- our eyetracking dataset (for langmod/NEAT)
  if params.TASK ~= 'combined' and params.TASK ~= 'langmod' then
      crash()
  end
  DATASET_DIR = "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/questions/"
  readDict.corpusDir = "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/questions/numerified-anonymous-joint/texts/"
  readDict.dictLocation = "/afs/cs.stanford.edu/u/mhahn/num2CharsNoEntities-deepmind-joint" --JOINT
  LIST_OF_FILES_TO_READ =  "/afs/cs.stanford.edu/u/mhahn/mhahn_files/ed1516/experiments/corpus/forEvaluation/fileList.txt"
elseif DATASET == 17 then -- qa -- the SAMPLE from the joint dataset (100 texts from CNN+Dailymail)
  if params.TASK ~= 'combined' and params.TASK ~= 'langmod' then
      crash()
  end
  assert(DOING_EVALUATION_OUTPUT)
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/joint/training/noentities-sample/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/joint/training/noentities/texts/"
  readDict.dictLocation = "/afs/cs.stanford.edu/u/mhahn/num2CharsNoEntities-deepmind-joint" --JOINT
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/deepmind-sample-dev.txt"
elseif DATASET == 18 then -- qa
  print("CNN Dev set")
  if params.TASK ~= 'neat-qa' then
      crash()
  end
  DATASET_DIR = "/u/scr/mhahn/deepmind-qa/cnn/dev/noentities/"
  readDict.corpusDir = "/u/scr/mhahn/deepmind-qa/cnn/dev/noentities/"
  readDict.dictLocation = "/u/scr/mhahn/num2CharsNoEntities-cnn"
  LIST_OF_FILES_TO_READ =  "/u/scr/mhahn/cnn-index-dev.txt"
  print("Please consider. Reading files from "..LIST_OF_FILES_TO_READ)
else
  crash()
  DATASET_DIR = "/afs/inf.ed.ac.uk/user/s15/s1582047/repo/mhahn_files/ed1516/lstm/lstm/corpus7/texts/"
  readDict.corpusDir = "/afs/inf.ed.ac.uk/user/s15/s1582047/repo/mhahn_files/ed1516/lstm/lstm/corpus7/texts/"
  readDict.dictLocation = "/afs/inf.ed.ac.uk/user/s15/s1582047/repo/mhahn_files/ed1516/lstm/lstm/corpus7/num2Chars"
  LIST_OF_FILES_TO_READ =  "/afs/inf.ed.ac.uk/user/s15/s1582047/repo/mhahn_files/ed1516/lstm/lstm/corpus7/files"
end

print("DATASET "..DATASET_DIR)
