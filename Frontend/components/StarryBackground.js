import { StarIcon } from "evergreen-ui";
import { useEffect,useState } from "react"

export default function StarryBackground({color,zIndex}){
    const [stars,setStars] = useState([]);
    
    useEffect(()=>{
        var numStars = 20 + Math.random()*5;
        var arr = []
        for(let i = 0; i < numStars;i++){
            arr.push(<StarIcon 
                className="starInSky" 
                top={`${Math.random()*100}%`} 
                left={`${Math.random()*100}%`} 
                position='absolute' 
                zIndex={zIndex} 
                size={5+Math.random()*10} 
                color={color} 
                animationDelay={`${1*Math.random()}s`
            }></StarIcon>)
        }
        setStars(arr);
    },[])

    return <div className="starryBackground">
        {stars}
    </div>
}