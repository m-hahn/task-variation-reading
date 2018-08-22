qa = {}
qa.__name = "qa"

--print(qa)

function qa.getFromAnswer(data, index,token)
   local answer = data[index].answer
--   print(numbersToEntityIDs)
  -- print(answer)
   if #answer >= token then
       return answer[token] --note this has already been converted to an answer ID
   else
       return 1
   end
end

function qa.getFromText(data, index,token)
   local text = data[index].text
   if #text >= token then
       return text[token]
   else
       return 1
   end
end

function qa.getFromQuestion(data, index,token)
   local question = data[index].question
   if #question >= token then
       return question[token]
   else
       return 1
   end
end

function qa.buildAnswerTensor(data, startIndex, endIndex)
    return buildInputTensorsForSubcorpus(data, startIndex, endIndex, qa.getFromAnswer, params.seq_length, torch.LongTensor)[1]
end

local function printStuffForQA(perp, actor_output, since_beginning, epoch, numberOfWords)

            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..readChunks.corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(experimentNameOut)
            print(params) 

            print('Acc '..((0.0+qaCorrect) / (qaIncorrect + qaCorrect)))
   
            for l = 1, 1 do
               print("....")
               print(perp[l])
               --[[for j=1,textLengths[l] do
                  io.write((readDict.chars[getFromText(corpus,l,j)]))--..'\n')
                  io.write("  "..attention_decisions[j][l].."  "..attention_scores[j][l][1].."\n")
               end]]
--- HUHUQA
               local answerID = getFromAnswer(corpus,l,1)
               if answerID == nil then
                    answerID = 1
               end
               print(answerID)
               --print(answer_output)
               --print(answer_output[l])
               print(answer_output[l][answerID])
               print(math.exp(-answer_output[l][answerID]))

            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)
end




