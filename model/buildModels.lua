-- ~/local/bin/th main-attention.lua 1 false true 60 50 1000 10000 4.5 true 0.7 100 0.0001 20 false pg-test-combined-50-1000-0.7-100-R-5a0-5.0-entities-entropy5.0-1.0-combined-real-full a0-4.5-entities-entropy5.0-1.0-combined-real-full-re10 combined true 8 true 5.0 none", "pg-test-combined-50-1000-0.7-100-R-4.5a0-4.5-entities-entropy5.0-1.0-combined-real-full-re10

ENTROPY_WEIGHT = "4.0"

NEW_ENCODER = true

   if NEW_ENCODER then
       command = {"1","false","true","60","50","1000","10000","4.1","true","0.7","100","0.0001","20","false","pg-test-combined-50-1000-0.7-100encoder-1","a0-5.0-entities-entropy5.0-1.0encoder-1-full-re10","combined","true","8","true",ENTROPY_WEIGHT,"none","false"}
   else
       command = {"1","false","true","60","50","1000","10000","5.0","true","0.7","100","0.0001","20","false","pg-test-combined-50-1000-0.7-100-R-5a0-5.0-entities-entropy5.0-1.0-combined-real-full","a0-5.0-entities-entropy5.0-1.0-combined-real-full-re10","combined","true","8","true",ENTROPY_WEIGHT,"none","false"}
   end

-- have started
-- CUDA_VISIBLE_DEVICES=0 ~/local/bin/th buildModels.lua 5 10 1
-- CUDA_VISIBLE_DEVICES=3 ~/local/bin/th buildModels.lua 1 4 2

local myarg = arg




--note = "-nopreview"
note = ""

local function buildModels(abl, weight)
     for i=myarg[1]+0, myarg[2]+0 do
        command[1] = myarg[3] --process ID
        command[8] = weight
        if NEW_ENCODER then
           command[16] = "a0-"..weight.."-entities-entropy"..ENTROPY_WEIGHT.."-1.0encoder-1-"..abl..note.."-re"..i
        else
           command[16] = "a0-"..weight.."-entities-entropy"..ENTROPY_WEIGHT.."-1.0-combined-real-"..abl..note.."-re"..i
        end
        command[22] = abl
        print(command)
        arg = command
        dofile("main-attention.lua", command)
     end
end

buildModels("rs", "5.0") -- no context
buildModels("full", "5.0")
buildModels("ir", "5.0") -- only surp
buildModels("s", "5.0") -- no surp

if false then
  buildModels("rs", "4.5") -- no context
  buildModels("ir", "4.5") -- only surp
  buildModels("full", "4.5")
  buildModels("s", "5.0") -- no surp
end
















