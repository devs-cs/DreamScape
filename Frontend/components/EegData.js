import { useContext, useState, useEffect} from "react";
import Loading from "./Loading";
import PContext from "../config/context";

export default function EegData(){
    const n = 5;
    const {nums,setStage,setTexts,scrollPositions} = useContext(PContext);
    const [data,setData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingProgress,setLoadingProgress] = useState(0); //purely used for loading bar;

    useEffect(()=>{
        if(nums&&nums.length>0&!nums.includes(-1)&&data.length==0){
            setIsLoading(true);
        }
    },[nums])

    useEffect(()=>{
        console.log(isLoading);
        if(isLoading) generateData();
    },[isLoading])

    const generateData = () => {
        setIsLoading(true);
        generateSingleData(0,nums,[]);
    }

    const renderData = () => {
        var a = [];
        if(data){
            a.push(<li className="labels">
                <label style={{height: "30px"}}></label>
                <label>EEG Data</label>
                <label>Embedding</label>
                <label>Image</label>
            </li>)
            for(let i = 0;i<n;i++){
                let d = data[i];
                if(!d) continue;
                a.push(<li>
                    <label style={{height: "30px", fontWeight: "normal", fontSize: "15px"}}>Image {i}</label>
                    <img className="graphImage" src={`data:image/jpeg;base64,${d.eeg_img}`}></img>
                    <img className="graphImage" src={`data:image/jpeg;base64,${d.embed_img}`}></img>
                    <img className="graphImage" src={`data:image/jpeg;base64,${d.gen_img}`}></img>
                </li>)
            }
        }
        return a;
    }

    const generateSingleData = async (i,allIndices,res) => {
        try{
            setLoadingProgress((i+1)/allIndices.length);
            if(i>=allIndices.length) {
                setIsLoading(false);
                setData(res);
                return;
            }
            const data = {"item_idx": Number(allIndices[i])+1}
            const url = "https://2d02-67-134-204-45.ngrok-free.app/data";

            var r = await fetch(url,{
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            })
            var d = await r.json();
            console.log(d);
            generateSingleData(i+1,allIndices,[...res,d]);
        }catch(e){
            console.error(e);
            generateSingleData(i+1,allIndices,[...res]);
        }
    }

    const continueFunc = () => {
        setStage(3);
        setTexts(data.map(d=>d.desc))
    }

    return <div id="eegDataContainer" style={{width: "100%", height: "100%", justifyContent: "center"}} className="fc" >
        {isLoading?<Loading progress={loadingProgress} text="Generating Images and Data"></Loading>:<ul className="fc">
            {nums&&nums.length>0&&<ul id="dataGrid">{renderData()}</ul>}
           {!isLoading&&<button className="sb" onClick={()=>continueFunc()}>Continue</button>} 
        </ul>}
    </div>
}