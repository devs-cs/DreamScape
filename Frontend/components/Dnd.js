import { Button, DoubleChevronLeftIcon, DoubleChevronRightIcon } from 'evergreen-ui';
import { useContext, useEffect, useState } from 'react'
import PContext from '../config/context';

export default function Dnd() {
  const n = 10;
  const l = 5;
  const [startingIndex,setStartingIndex] = useState(0);
  const [canContinue,setCanContinue] = useState(false);
  const urls = ['image1.jfif','image2.jfif','image3.jfif','image4.jfif','image5.jfif',"image6.jfif","image7.jfif","image8.jfif","image9.jfif","image10.jfif"];
  const [inBucket,setInBucket] = useState(Array.from({ length: l }, (v, i) => -1)); // all -1 at first;
  const {setNums,scrollPositions,setStage} = useContext(PContext);

  const generateList = () => {
    var res = [];
    for(let i = 0;i<n;i++){
      if (!inBucket.includes(i)) res.push(<li
        draggable="true"
        onDragStart={(e)=>{
          e.dataTransfer.setData("imageNum",i);
        }}
        style={{backgroundImage: `url(/images/${urls[i]})`, backgroundSize: "cover", backgroundPosition: "center"}}
        >
      </li>)
    }
    return res;
  }

  useEffect(()=>{
    var isAllFilled = true;
    inBucket.forEach(a=>{
      if(a==-1) isAllFilled = false;
    })
    setCanContinue(isAllFilled);
    console.log(isAllFilled);

  },[inBucket])


  const handleDrop = (e,i) => {
    const newInBucket = [...inBucket];
    newInBucket[i] = e.dataTransfer.getData("imageNum");
    setInBucket(newInBucket);
  }

  const generateBuckets = () => {
    var res = [];
    for(let i = 0;i<l;i++){
      res.push(
      <div>
        <li 
          onDrop={(e)=>handleDrop(e,i)}
          onDragOver = {(e)=>{e.preventDefault();}} 
          style={{backgroundImage: inBucket[i]==-1?"none":`url(/images/${urls[inBucket[i]]})`, backgroundSize: "cover", backgroundPosition: "center"}}
        >
          {inBucket[i]==-1&&<>
            <h6>Image {i + 1}</h6>
            <p>Pick Your Dream Sequence!</p>
          </>}
        </li>
      </div>)
    }
    return res;
  }

  const continueToNext = () => {
    setNums(inBucket);
    setStage(2);
  }


  return (
    <section className="dnd">
      <ul className='horizList scrollList'>
        {generateList().slice(startingIndex,startingIndex+l)}
      </ul>
      <div className='leftRightArrows'>
        {startingIndex>0&&<button className='leftArrow' onClick={()=>setStartingIndex(startingIndex-1)}><DoubleChevronLeftIcon></DoubleChevronLeftIcon></button>}
        {startingIndex<n-l&&<button className="rightArrow" onClick={()=>setStartingIndex(startingIndex+1)}><DoubleChevronRightIcon></DoubleChevronRightIcon></button>}
      </div>
      <ul className="horizList">
        {generateBuckets()}
      </ul>
      {canContinue&&<button className="sb" style={{marginTop: "30px"}} onClick={(e)=>{
        continueToNext()}
      }>Use This Sequence</button>}
      {/* <img src={`data:image/jpeg;base64,${image}`} className='testImage'></img> */}

    </section>
  )
}