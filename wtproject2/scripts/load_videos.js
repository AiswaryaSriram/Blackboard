function construct_video_col(loc,videotitle,videodesc,card_type)
{
	col=document.createElement("div");
	col.className="col-md-9";

	card_chart=document.createElement("div")
	card_chart.className="card card-chart";
	col.appendChild(card_chart);

	card_body=document.createElement("div");
	card_body.className="card-body";
	card_header=document.createElement("div");
	card_header.className="card-header card-header-"+card_type;
	card_chart.appendChild(card_header);
	card_chart.appendChild(card_body);

	embed=document.createElement("div");
	embed.className="embed-responsive embed-responsive-1by1"
	card_header.appendChild(embed);

	video=document.createElement("video");
	video.className="embed-responsive-item";
	video.controls=true;
	embed.appendChild(video);


	source=document.createElement("source");
	source.setAttribute('src', loc);
	source.setAttribute("type","video/mp4");
	video.appendChild(source);
	

	h4body=document.createElement("h4");
	h4body.className="card-title";
	h4body.innerHTML=videotitle;
	pbody=document.createElement("p");
	pbody.className="card-category";
	pbody.innerHTML=videodesc;
	card_body.appendChild(h4body);
	card_body.appendChild(pbody);
	
	return col;
	

}
/*scrollAmt=100;
count=0;
function gettext()
{
	xhr=new XMLHttpRequest();
	scroll=document.body.scrollTop||document.documentElement.scrollTop;
	if(scroll>scrollAmt){
		scrollAmt=scroll;
		xhr.onreadystatechange=showChunk;
		xhr.open("GET","getChunk.php?count="+count++,true);
		xhr.send();
	}
}
function showChunk()
{
	console.log(xhr.status)
	if(xhr.readyState==4 && xhr.status==200)
	{
		console.log("hello");
		document.getElementById("content").innerHTML+=xhr.responseText;
							
				
	}				
			
}
window.onscroll=getChunk;
*/
m=document.getElementById("main_content");

json=JSON.parse(localStorage.getItem("videos"));
colours=["success", "warning","danger"];
i=0
row=document.createElement("div");
row.className="row";
m.appendChild(row);

link=document.createElement("a");
link.innerHTML="Link to text book";
link.href="textbook.html";
/*
link.addEventListener("click",gettext,false);*/
row.appendChild(link);
for(x in json)
{
	row=document.createElement("div");
	row.className="row";
	loc="videos/"+x;
	title=json[x]['videotitle'];
	desc=json[x]['videodesc'];

	col=construct_video_col(loc,title,desc,colours[i]);
	i=(i+1)%3;
	row.appendChild(col)
	m.appendChild(row)
			//alert(loc + "   "+title);
}

