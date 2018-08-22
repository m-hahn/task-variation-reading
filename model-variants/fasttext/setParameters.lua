
-- 1 GPU [NOW: PROCESS ID]
-- 2 doing evaluation output?
-- 3 LOAD
-- 4 batch size
-- 5 seq length
-- 6 rnn size
-- 7 vocab size
-- 8 total attentions weight
-- 9 use attention network?
-- arg[10] is learning rate
-- embeddings_dimensionality = arg[11]
-- arg[12] is lr_att
-- arg[13] is minus ATTENTION_VALUES_BASELINE
-- arg[14] is whether it is reloaded from a previous reloaded version (in which case the attention network will be taken over)
-- arg[15] the file to be loaded (tacitly assuming that dimensions will match)
-- arg[16] is a suffix to the files that are written
-- arg[17] the task
-- arg[18] whether it should do training at all
-- arg[19] the number of the corpus
-- arg[20] ATTENTION_WITH_EMBEDDINGS
-- arg[21] ENTROPY WEIGHT
-- arg[22] ABLATION
-- arg[23] overwrite model? (true or false)
-- arg[24] external attention source
-- arg[25] condition (optional)
-- arg[26] CONDITION_BIAS (optional)



--SCRATCH_PATH = "/jagupard11/scr1/mhahn/"
SCRATCH_PATH = "/u/scr/mhahn/"

CONDITION_FROM_SETPARAMETERS = nil
if (#arg > 24) then
   CONDITION_FROM_SETPARAMETERS = arg[25]
end
--print(arg[25])
print("SETTING CONDITION")
print(CONDITION_FROM_SETPARAMETERS)
--------------------------
--------------------------
DOING_DEBUGGING = false --false
print("DOING DEBUGGING? ")
print(DOING_DEBUGGING)

if true then
  USE_PREDICTION_FOR_ATTENTION = false
  print("WARNING 3317 Hard-coded whether to use baseline")
  if true or CONDITION_FROM_SETPARAMETERS == "mixed" then
    USE_BIDIR_BASELINE = false
    USE_SIMPLE_BASELINE = true
  else
    USE_BIDIR_BASELINE = true --true--false
    USE_SIMPLE_BASELINE = false
  end
else
  USE_PREDICTION_FOR_ATTENTION = true
  USE_BIDIR_BASELINE = false
  USE_SIMPLE_BASELINE = false

--  PERCENTAGE_OF_DATA = 100
end

if arg[17] ~= "combined" and arg[17] ~= "neat-qa" and (USE_BIDIR_BASELINE == true or USE_SIMPLE_BASELINE==true) then
   USE_BIDIR_BASELINE = false
   USE_SIMPLE_BASELINE = false
   print("WARNING overriding USE_BIDIR_BASELINE, USE_SIMPLE_BASELINE")
--   crash()
end




print("USE_BIDIR_BASELINE  ")
print(USE_BIDIR_BASELINE)

print("USE_SIMPLE_BASELINE  ")
print(USE_SIMPLE_BASELINE)


USE_BASELINE_NETWORK = false


PRINT_MODEL_PERIOD = 500 --500 -- 1000



if USE_PREDICTION_FOR_ATTENTION and (USE_BIDIR_BASELINE or USE_SIMPLE_BASELINE) then
   crash()
end

--------------------------
--------------------------

--------------------------
--------------------------

NLL_TO_CHANGE_ATTENTION = 0.00000001

meanNLL = 10000
meanTotalAtt = 0

--------------------------
--------------------------

corpus_name = nil --"pg50665.txt" --"pg50665.txt" --"hlm.txt"

if arg[3] == 'false' then
   arg[3] = false
end
LOAD = arg[3] and true

--------------------------
--------------------------

REWARD_DIFFERENCE_SCALING = 1

FIXED_ATTENTION = 1.0
BASE_ATTENTION = 0.6


function makeBoolean(string)
   if string == "false" then
     return false
   elseif string == "true" then
     return true
   else
     print(string)
     crash()
   end
end

print(arg)

DOING_EVALUATION_OUTPUT = makeBoolean(arg[2])


OVERWRITE_MODEL = makeBoolean(arg[23])


if arg[24] == nil then
  print("WARNING arg[24] is nil")
  arg[24] = 'fixed'
elseif string.sub(arg[24], 1, 14) == "NUMERICAL_FILE" then
 -- 3: FIXNO
-- 4: WordFreq
-- 5: WLEN
-- 9: surprisal
    NUMERICAL_VALUES_COLUMN = string.sub(arg[24], 15) + 0.0
    arg[24] = "NUMERICAL_FILE"
    print("Numerical Values Column: "..NUMERICAL_VALUES_COLUMN) 
elseif arg[24] ~= "WLEN" and arg[24] ~= "fixed" then
  print(arg[24])
  crash()
end

-- 1 pg50665.txt 20 30 100 8000
params = {process_id = arg[1]+0,
                batch_size=arg[4]+0,
                seq_length=arg[5]+0,
                --layers=2,
                --decay=2,
                rnn_size=arg[6]+0,
                baseline_rnn_size=20,
                --dropout=0,
                init_weight=0.05,
                lr=((arg[10]+0) + 0.0), -- 0.01
                vocab_size=arg[7]+0,
                --max_epoch=4,
                --max_max_epoch=13,
                max_grad_norm=5,
                lr_att =(arg[12]+0.0),
                lr_momentum = 0.95, --0.9, --was 0.4 for all experiments
                embeddings_dimensionality = arg[11] + 0,
                ATTENTION_VALUES_BASELINE = - (arg[13] + 0.0),
                TOTAL_ATTENTIONS_WEIGHT = arg[8]+0,
                EXTERNAL_ATTENTION_SOURCE = arg[24], --'NUMERICAL_FILE' --'WLEN' --'fixed'
                gpu_number = 1,
                TASK = arg[17],
                ATTENTION_WITH_EMBEDDINGS = makeBoolean(arg[20]),
                INCLUDE_NUMERICAL_VALUES = (arg[24] == "NUMERICAL_FILE"),
                ablation = arg[22],
                ENTROPY_PENALTY = arg[21] + 0.0} --hqhq

params.CONDITION_BIAS = 0.0
if (#arg > 25) then
  params.CONDITION_BIAS = arg[26]+0.0
end


evaluationAccumulators = {reconstruction_loglikelihood = 0,
                                lm_loglikelihood = 0,
                                numberOfTokens = 0}



use_attention_network = nil
train_attention_network = nil
train_autoencoding = nil

if arg[9] == 'false' then
   arg[9] = false
end
if arg[9] then
  use_attention_network = true
  train_attention_network = true
  train_autoencoding = false
else
  use_attention_network = false
  train_attention_network = false
  train_autoencoding = true
end   
if train_attention_network and (not use_attention_network) then
   crash()
end

if params.TASK == 'qa' then
   qaIncorrect = 0
   qaCorrect = 0
end

if arg[18] == 'false' then
   DO_TRAINING = false
elseif arg[18] == 'true' then
   DO_TRAINING = true
else
   crash()
end

print("DO TRAINING?: "..tostring(DO_TRAINING))

if arg[14] == 'false' then
   arg[14] = false
end
IS_CONTINUING_ATTENTION = arg[14]


fileToBeLoaded = arg[15]

suffixForSaving = arg[16]



experimentName = "pg-test-"..params.TASK.."-"..params.seq_length.."-"..params.rnn_size.."-"..params.lr.."-"..params.embeddings_dimensionality
experimentNameOut = experimentName

if LOAD then
  experimentNameOut = experimentNameOut.."-R-"..params.TOTAL_ATTENTIONS_WEIGHT
end

if IS_CONTINUING_ATTENTION then
   experimentName = experimentNameOut
   experimentNameOut = experimentNameOut.."-R2"
end

if suffixForSaving ~= nil then
   experimentNameOut = experimentNameOut..suffixForSaving
end

--SCRATCH_PATH.."cuny-plots/"..
fileStats = io.open(experimentNameOut..'-nll-stats', 'w')
--fileStatsAccuracy = io.open(experimentNameOut..'-acc-stats', 'w')
fileStatsReward = io.open(experimentNameOut..'-reward-stats', 'w')
fileStatsFixations = io.open(experimentNameOut..'-fix-stats', 'w')

fileStatsLrAtt = io.open(experimentNameOut..'-lratt-stats', 'w')
fileStatsEntropyW = io.open(experimentNameOut..'-entw-stats', 'w')
fileStatsTotAttW = io.open(experimentNameOut..'-attw-stats', 'w')


print("Printing stuff to "..experimentNameOut)



function transfer_data(x)
--for i=1,100 do
--print("WARNING: USING CPU!!!!")
--end
--  return x
  return x:cuda()
end

state_train, state_valid, state_test  = nil
model = {}
paramx, paramdx = nil

------------------------

DATASET = arg[19] + 0


-- QA PARAMETERS
MAX_LENGTH_Q_FOR_QA = 30 -- 20--20 -- hihi
MAX_LENGTH_T_FOR_QA = 500 --500--500 --hihi

if DATASET == 5 or DATASET == 12 or DATASET == 15 or DATASET == 18 then
  NUMBER_OF_ANSWER_OPTIONS = 600 --600
elseif DATASET == 6 then
  NUMBER_OF_ANSWER_OPTIONS = 8 --600
else
 assert(params.TASK ~= 'neat-qa')
 print("Warning 71")
end

EPOCHS_NUMBER =  2 --4 --2 --20000000 --20
if params.TASK == "combined" then
   EPOCHS_NUMBER = 1
end

  if params.TASK == "combined" then
     PERCENTAGE_OF_DATA = 30
  else
     PERCENTAGE_OF_DATA = 98
  end
print("EPOCHS_NUMBER: "..EPOCHS_NUMBER)
if DOING_EVALUATION_OUTPUT then
   CREATE_RECORDED_FILES = true
   if (not CREATE_RECORDED_FILES) then
     PERCENTAGE_OF_DATA = 20
     print("Warning: setting PERCENTAGE_OF_DATA at 32154")
   end
   if DATASET == 18 then
      CREATE_RECORDED_FILES = false
      PERCENTAGE_OF_DATA = 99
   end
   print("Creating recoded files? "..tostring(CREATE_RECORDED_FILES))

else
   CREATE_RECORDED_FILES = false
end

print("PERCENTAGE OF DATA "..PERCENTAGE_OF_DATA)

