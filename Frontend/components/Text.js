import { useContext,useState,useEffect } from "react";
import PContext from "../config/context";
import autoscroll from "../config/autoscroll";
import Loading from "./Loading";
import Spinner from "./Spinner";

export default function Text(){
    const n = 5;
    const [data,setData] = useState("");
    const [isLoading,setIsLoading] = useState(false);
    const {stage,setStage,setStory,texts} = useContext(PContext);

    useEffect(()=>{
        if(stage == 3 && texts && texts.length > 0 ){
            setIsLoading(true);
        }
    },[stage])

    useEffect(()=>{
        if(isLoading) generateData();
    },[isLoading])

    const generateData = async () => {
        try{
            console.log(texts);
            const jsonData = {"data": texts};
            const url = "https://2d02-67-134-204-45.ngrok-free.app/text_to_story";

            var r = await fetch(url,{
                method: "POST",
                headers: {
                    "Content-type": "application/json"
                },
                body: JSON.stringify(jsonData)
            })
            var d = await r.text();
            setData(d);
            setStory(d);
            setIsLoading(false);
        }catch(e){
            console.error(e);
        }
    }


    const continueFunc = () =>{
        setStage(4);
    }

    const renderTexts = () => {
        console.log(texts);
        var arr = [];
        for(let i = 0;i<texts.length;i++){
            arr.push(<li>
                <label>Image {i+1}:</label> {texts[i]}
            </li>)
        }
        return arr;
    }

    return <div id="textStoryContainer" className="fc">
        <ul className="textsList">{renderTexts()}</ul>
        {isLoading?<div className="fc gs"><label>Generating</label><Spinner></Spinner><label>Story</label></div>:<div className="story">{data}</div>}
        {!isLoading&&<button className="sb" onClick={()=>continueFunc()} style={{marginTop: "30px"}}>Continue</button>}
    </div>
} 