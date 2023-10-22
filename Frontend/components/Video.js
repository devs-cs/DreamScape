import { useContext,useState,useEffect, useRef } from "react";
import PContext from "../config/context";
import { PlayIcon } from "evergreen-ui";

export default function Video(){
    const usePreVideo = false;
    const maxCount = 10;
    const interval = 2000; //in ms
    const {stage,story} = useContext(PContext);

    const [start,setStart] = useState(0); // only on 1 
    const [encoding,setEncoding] = useState("");
    const encodingRef = useRef("");


    
    const generateDataFirstTime = async (story,count) => {
        try{
            const jsonData = {"start": true,"Text": story};
            const url = "https://2d02-67-134-204-45.ngrok-free.app/text_to_images";

            var r = await fetch(url,{
                method: "POST",
                headers: {
                    "Content-type": "application/json"
                },
                body: JSON.stringify(jsonData)
            })
            var d = await r.json();


            if(d.data) setEncoding(d["data"]);


        }catch(e){
            console.error(e);
        }finally{
            setTimeout(()=>{
                generateDataAfter(count+1);
            },interval)
        }
    }

    const generateDataAfter = async (count) => {
        console.log(count);
        if(count > maxCount) return;
        try{
            const jsonData = {"start": false};
            const url = "https://2d02-67-134-204-45.ngrok-free.app/text_to_images";

            var r = await fetch(url,{
                method: "POST",
                headers: {
                    "Content-type": "application/json"
                },
                body: JSON.stringify(jsonData)
            })
            var d = await r.json();
            console.log(d);

            if(d.img) {
                console.log(d.data==encoding);
                setEncoding(String(d.data));
            }
        }catch(e){
            console.error(e);
        }finally{
            setTimeout(()=>{
                generateDataAfter(count+1);
            },interval)
        }
    }

    useEffect(()=>{
        if(start==1) {
            setStart(3);
            generateDataAfter(0);
        }else if(start == 2) {
            setStart(3);
            generateDataFirstTime(story,0);
        }
    },[start])

    useEffect(()=>{
        console.log(encoding.substring(0,10))
    },[encoding])




    return <div id="videoContainer" className="fc">
        {usePreVideo&&stage==4&&<video width="530" height="300" controls className="preVideo">
            <source src="/video/video1.mp4" type="video/mp4">
            </source>
        </video>}
        {!usePreVideo&&<div className="fc">
            <div className="imageSequenceContainer">{/*encoding.substring(4000,4001)*/}
            {encoding?<img className="imageSeq" src={`data:img/jpeg;base64,${encoding}`}></img>:<div className="imageSeq">Sequence of images will play</div>}
            </div>
            <div>
                <button className="sb" onClick={()=>setStart(1)}><PlayIcon></PlayIcon></button>
                <button className="sb" style={{backgroundColor: "red", marginLeft: "10px"}} onClick={()=>setStart(2)}><PlayIcon></PlayIcon></button>
            </div>
        </div> }
    </div>
}