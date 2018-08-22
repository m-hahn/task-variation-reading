getParams = {}

TEMP = {}

function getParamFromFile(filename)
   if pcall(function ()    io.input(filename)
     TEMP.param = nil
--     print(filename)
            local t = io.read("*all")
  --          print(t)
            io.input():close()
            for line in string.gmatch(t, "[^\n]+") do
              if line:len() > 0 then
                TEMP.param= line+0.0
              end
            end
      end) then
      print(filename.."  "..TEMP.param)
      return TEMP.param
   else
      print("ERROR "..filename)
      return nil
   end
end


function getBaselineFromFile()
     local filename = "/u/scr/mhahn/baseline-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.ATTENTION_VALUES_BASELINE = -result
     end
end

function getL2RegFromFile()
     local filename = "/u/scr/mhahn/l2-reg-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.l2_regularization = result
     end
end 




function getAttentionFromFile()
     local filename = "/u/scr/mhahn/attention-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       FIXED_ATTENTION = result
     end
 



--[[            io.input("/disk/scratch2/s1582047/attention-"..arg[1])
            local t = io.read("*all")
            io.input():close()
            for line in string.gmatch(t, "[^\n]+") do
              if line:len() > 1 then
                FIXED_ATTENTION = line+0.0
              end
            end]]
end

function getLearningRateFromFile()
     local filename = "/u/scr/mhahn/lr-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.lr = result
     end
end

function getAttentionLearningRateFromFile()
     local filename = "/u/scr/mhahn/lr-att-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.lr_att = result
     end
end

function getEntropyWeightFromFile()
     local filename = "/u/scr/mhahn/entropy-weight-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.ENTROPY_PENALTY = result
     end
end




function getTotalAttentionWeightFromFile()
     local filename = "/u/scr/mhahn/total-att-weight-"..arg[1]
     local result = getParamFromFile(filename)
     if result ~= nil then
       params.TOTAL_ATTENTIONS_WEIGHT = result
     end

--[[            io.input("/disk/scratch2/s1582047/total-att-weight-"..arg[1])
            local t2 = io.read("*all")
            io.input():close()
            for line in string.gmatch(t2, "[^\n]+") do
              if line:len() > 1 then
                params.TOTAL_ATTENTIONS_WEIGHT = line+0.0
              end
            end]]
end
